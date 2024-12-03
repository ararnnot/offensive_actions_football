
import torch
from   tqdm import tqdm

class Lipschitz_Vector():
    
    def __init__(self, X, Y, distance_matrix_function,
                 compute_K = True, device = 'available',
                 precision = 'single', chunk_size = 2**8):
        """
        Args:
            X: Tensor of known points in the domain of f.
            Y: Tensor of known function values on X, can be vector-valued (using max K).
            compute_K: If True, compute the Lipschitz constant K.
            device: Device to use ('available', 'cuda' or 'cpu').
            precision: Precision of the computations ('half', 'single' or 'double').
            chunk_size: Size of the chunks to divide the computations (check for memory).
        """
        
        if device == 'available':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f'Using {self.device} device.')
        
        if precision == 'half':
            self.dtype = torch.float16
        elif precision == 'single':
            self.dtype = torch.float32
        elif precision == 'double':
            self.dtype = torch.float64
        
        self.chunk_size = chunk_size  
        self.distance_matrix_function = distance_matrix_function
        
        self.X = torch.tensor(X, device = self.device, dtype = self.dtype)
        self.Y = torch.tensor(Y, device = self.device, dtype = self.dtype)
        
        if compute_K:
            self.compute_constant_vector()
                         
    def compute_constant_max(self, show_progress = True):
        """
            Compute the Lipschitz constant K.
            If Y is a vector, K = max_l Lispchitz constant of the l-th component.
        """
        
        # if dist(x, x') = 0, we set it to 1, since f(x) - f(x') = 0 (no affect)
        distances = self.distance_matrix_function(self.X, self.X,
                                                  device = self.device, dtype = self.dtype)
        distances = torch.where(distances == 0, torch.tensor(1, device = self.device), distances)
        
        # Divided in chucks to avoid memory issues
        CHUNK_SIZE = self.chunk_size
        K = torch.tensor(0, device = self.device, dtype = self.dtype)
        for i in tqdm(range(0, len(self.X), CHUNK_SIZE),
                      desc = 'Computing Lipschitz constant',
                      disable = not show_progress):
            if i == 0:
                arr = torch.abs(self.Y[i:i+CHUNK_SIZE].unsqueeze(1) - self.Y.unsqueeze(0)) / \
                            distances[i:i+CHUNK_SIZE].unsqueeze(-1)
                mem = arr.nelement() * arr.element_size()
            this_K = torch.max(torch.abs(self.Y[i:i+CHUNK_SIZE].unsqueeze(1) - self.Y.unsqueeze(0)) / \
                        distances[i:i+CHUNK_SIZE].unsqueeze(-1))
            K = torch.max(K, this_K)
            
        self.K = torch.full((self.Y.shape[1], ), K,
                            device = self.device, dtype = self.dtype)
            
        if show_progress and self.device == 'cuda':
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            print(  f"Memory usage per chunck: " + \
                    f"{mem/1024**3:.2f} GB of {total_mem/1024**3:.2f} GB available (total).")
            print(f"Lipschitz constant K: {self.K}.")
        
    def compute_constant_vector(self, show_progress = True):
        """
            Compute the Lipschitz constant K.
            If Y is a vector, K is computed pointwise for each component.
        """
        
        # if dist(x, x') = 0, we set it to 1, since f(x) - f(x') = 0 (no affect)
        distances = self.distance_matrix_function(self.X, self.X,
                                                  device = self.device, dtype = self.dtype)
        distances = torch.where(distances == 0, torch.tensor(1, device = self.device), distances)
        
        # Divided in chucks to avoid memory issues
        CHUNK_SIZE = self.chunk_size
        K = torch.zeros(self.Y.shape[1], device = self.device, dtype = self.dtype)
        for i in tqdm(range(0, len(self.X), CHUNK_SIZE),
                      desc = 'Computing Lipschitz constant',
                      disable = not show_progress):
            if i == 0:
                arr = torch.abs(self.Y[i:i+CHUNK_SIZE].unsqueeze(1) - self.Y.unsqueeze(0)) / \
                            distances[i:i+CHUNK_SIZE].unsqueeze(-1)
                mem = memory_usage_bytes = arr.nelement() * arr.element_size()
            this_K = (torch.abs(self.Y[i:i+CHUNK_SIZE].unsqueeze(1) - self.Y.unsqueeze(0)) / \
                        distances[i:i+CHUNK_SIZE].unsqueeze(-1)
                    ).max(dim = 1)[0].max(dim = 0)[0]
            K = torch.max(K, this_K)
            
        self.K = K
            
        if show_progress and self.device == 'cuda':
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            print(  f"Memory usage per chunck: " + \
                    f"{mem/1024**3:.2f} GB of {total_mem/1024**3:.2f} GB available (total).")
            print(f"Lipschitz constant K: {self.K}.")        

    def mcshane_whitney_extension(self, new_X, show_progress = True):
        """
        McShane extension of a function Y (given on X) to new points new_X.
        Args:
            new_X: Tensor of points where we want to extend the function.
        Returns:
            Tensor of extended function values at points in x.
        """
        
        new_X = torch.tensor(new_X, device = self.device, dtype = self.dtype)
        distances = self.distance_matrix_function(new_X, self.X,
                                                  device = self.device, dtype = self.dtype)

        # Y and results are vectors
        mcshane_ext = torch.full((len(new_X), self.Y.shape[1]),
                                 float('nan'), device = self.device)
        whitney_ext = torch.full((len(new_X), self.Y.shape[1]),
                                 float('nan'), device = self.device)
        
        # Divided in chucks to avoid memory issues
        CHUNK_SIZE = self.chunk_size
        for i in tqdm(range(0, len(new_X), CHUNK_SIZE),
                      desc = 'Computing McShane-Whitney extension'):
            mcshane_ext[i:i+CHUNK_SIZE] = torch.min(
                self.Y.unsqueeze(0)
                    + self.K.unsqueeze(0).unsqueeze(1)
                    * distances[i:i+CHUNK_SIZE].unsqueeze(-1),
                dim = 1).values
            whitney_ext[i:i+CHUNK_SIZE] = torch.max(
                self.Y.unsqueeze(0)
                    - self.K.unsqueeze(0).unsqueeze(1)
                    * distances[i:i+CHUNK_SIZE].unsqueeze(-1),
                dim = 1).values
            if i == 0:
                arr = self.Y.unsqueeze(0) \
                        + self.K.unsqueeze(0).unsqueeze(1) \
                        * distances[i:i+CHUNK_SIZE].unsqueeze(-1)
                mem = arr.nelement() * arr.element_size()

        if show_progress and self.device == 'cuda':
            total_mem = torch.cuda.get_device_properties(self.device).total_memory 
            print(  f"Memory usage per chunck: " + \
                    f"{mem/1024**3:.2f} GB of {total_mem/1024**3:.2f} GB available (total).")

        extension = (mcshane_ext + whitney_ext) / 2
        return extension.cpu().numpy()

class Lipschitz_Real():
    
    def __init__(self, X, Y, distance_matrix_function,
                 compute_K = True, device = 'available',
                 precision = 'single', chunk_size = 2**15):
        """
        Args:
            X: Tensor of known points in the domain of f.
            Y: Tensor of known function values on X, can be vector-valued (using max K).
            compute_K: If True, compute the Lipschitz constant K.
            device: Device to use ('available', 'cuda' or 'cpu').
            precision: Precision of the computations ('half', 'single' or 'double').
            chunk_size: Size of the chunks to divide the computations (check for memory).
        """
        
        if device == 'available':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f'Using {self.device} device.')
        
        if precision == 'half':
            self.dtype = torch.float16
        elif precision == 'single':
            self.dtype = torch.float32
        elif precision == 'double':
            self.dtype = torch.float64
        
        self.chunk_size = chunk_size  
        self.distance_matrix_function = distance_matrix_function
        
        self.X = torch.tensor(X, device = self.device, dtype = self.dtype)
        self.Y = torch.tensor(Y, device = self.device, dtype = self.dtype)
        
        if compute_K:
            self.compute_constant()
                
    def compute_constant(self, show_progress = True):
        """
            Compute the Lipschitz constant K.
            If Y is a vector, K = max_l Lispchitz constant of the l-th component.
        """
        
        # if dist(x, x') = 0, we set it to 1, since f(x) - f(x') = 0 (no affect)
        distances = self.distance_matrix_function(self.X, self.X,
                                                  device = self.device, dtype = self.dtype)
        distances = torch.where(distances == 0, torch.tensor(1, device = self.device), distances)
        
        # Divided in chucks to avoid memory issues
        CHUNK_SIZE = self.chunk_size
        K = torch.tensor(0, device = self.device, dtype = self.dtype)
        for i in tqdm(range(0, len(self.X), CHUNK_SIZE),
                      desc = 'Computing Lipschitz constant',
                      disable = not show_progress):
            if i == 0:
                arr = torch.abs(self.Y[i:i+CHUNK_SIZE].unsqueeze(1) - self.Y.unsqueeze(0)) / \
                            distances[i:i+CHUNK_SIZE]
                mem = memory_usage_bytes = arr.nelement() * arr.element_size()
            this_K = torch.max(torch.abs(self.Y[i:i+CHUNK_SIZE].unsqueeze(1) - self.Y.unsqueeze(0)) / \
                        distances[i:i+CHUNK_SIZE])
            K = torch.max(K, this_K)
            
        self.K = K
            
        if show_progress and self.device == 'cuda':
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            print(  f"Memory usage per chunck: " + \
                    f"{mem/1024**3:.2f} GB of {total_mem/1024**3:.2f} GB available (total).")
            print(f"Lipschitz constant K: {self.K}.")
        
    def mcshane_whitney_extension(self, new_X, show_progress = True):
        """
        McShane extension of a function Y (given on X) to new points new_X.
        Args:
            new_X: Tensor of points where we want to extend the function.
        Returns:
            Tensor of extended function values at points in x.
        """
        
        new_X = torch.tensor(new_X, device = self.device, dtype = self.dtype)
        distances = self.distance_matrix_function(new_X, self.X,
                                                  device = self.device, dtype = self.dtype)

        # Y and results are vectors
        mcshane_ext = torch.full((len(new_X),),
                                 float('nan'), device = self.device)
        whitney_ext = torch.full((len(new_X),),
                                 float('nan'), device = self.device)
        
        # Divided in chucks to avoid memory issues
        CHUNK_SIZE = self.chunk_size
        for i in tqdm(range(0, len(new_X), CHUNK_SIZE),
                      desc = 'Computing McShane-Whitney extension'):
            mcshane_ext[i:i+CHUNK_SIZE] = torch.min(
                self.Y.unsqueeze(0)
                    + self.K
                    * distances[i:i+CHUNK_SIZE],
                dim = 1).values
            whitney_ext[i:i+CHUNK_SIZE] = torch.max(
                self.Y.unsqueeze(0)
                    - self.K
                    * distances[i:i+CHUNK_SIZE],
                dim = 1).values
            if i == 0:
                arr = self.Y.unsqueeze(0) \
                        + self.K \
                        * distances[i:i+CHUNK_SIZE]
                mem = memory_usage_bytes = arr.nelement() * arr.element_size()

        if show_progress and self.device == 'cuda':
            total_mem = torch.cuda.get_device_properties(self.device).total_memory 
            print(  f"Memory usage per chunck: " + \
                    f"{mem/1024**3:.2f} GB of {total_mem/1024**3:.2f} GB available (total).")

        extension = (mcshane_ext + whitney_ext) / 2
        return extension.cpu().numpy()
