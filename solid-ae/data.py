from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data       import DataLoader, random_split

#from transforms             import MinMaxNormalize

# Classe DataCarrier com:
#   - funcionalidade: baixar/carregar o dataset da memória
#   - funcionalidade: splittar o dataset
#   - funcionalidade: retornar dataloaders para treinamento

class MNISTDataCarrier():
    def __init__(self, split=(0.8, 0.1, 0.1), dataset_dir='datasets/', transformations=[ToTensor()]):
        assert sum(split) == 1, "Splits have to represent whole dataset (sum up to 1)"

        self.train_split     = split[0]
        self.val_split       = split[1]
        self.test_split      = split[2]
        self.dataset_dir     = dataset_dir
        self.transformations = transformations
        self.built           = False

    def _load_data(self):
        T = Compose(self.transformations)
        self.dataset = MNIST(self.dataset_dir, train=True, download=True, transform=T)
    
    def _split_data(self):
        train_size = int( self.train_split * len(self.dataset) )
        val_size   = int( self.val_split   * len(self.dataset) )
        test_size  = int( self.test_split  * len(self.dataset) )

        tr, v, t   = random_split(self.dataset, [train_size, val_size, test_size])
        self.train = tr
        self.val   = v
        self.test  = t

    def build(self):
        self._load_data()
        self._split_data()
        self.built = True
        

    def to_train(self, batch_size=32):
        if not self.built:
            self.build()
        TR = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        V  = DataLoader(self.val,   batch_size=batch_size, shuffle=False)
        T  = DataLoader(self.test,  batch_size=batch_size, shuffle=False)
        return TR, V, T













