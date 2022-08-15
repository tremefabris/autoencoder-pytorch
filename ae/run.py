import torch
import pytorch_lightning as pl
import matplotlib.pyplot  as     plt

from torchvision.datasets   import MNIST
from torch.utils.data       import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor

from transforms import MinMaxNormalize, Flatten
from model      import SimpleAutoencoder


def linearTransformation():
    linearTransformation = torch.nn.Linear(6, 8)
    print(linearTransformation.weight)

    x = torch.arange(6, dtype=torch.float32)
    print(x)

    y = linearTransformation(x)
    print(y)
    
    return None


def mnist_load():
    print("It's MNIST-ing time")

    mnist = MNIST(root='datasets/', train=True, download=True)
    img, label = mnist[0]
    print(type(img), type(label))

    plt.imshow(img, cmap='gray')
    plt.show()

    return mnist

def split():
    print("It's splitting time")

    mnist = MNIST(root='datasets/', train=True, download=True)
 
    # Splitting the dataset
    construction_size = int( 0.8 * len(mnist) )
    test_size         = len(mnist) - construction_size

    assert construction_size + test_size == len(mnist), "construction-test subsets need to have the same number of examples as original dataset"

    construction_raw, test_raw = random_split(mnist, [construction_size, test_size])
    print(len(construction_raw), len(test_raw))
    
    return construction_raw, test_raw


def preprocess():
    print("It's norm-ing time")

    # Composes transformations (ToTensor -> MinMaxNormalize)
    # MinMaxNormalize() and Flatten() are implemented in transforms.py (custom transformation)
    #
    #   Since PyTorch already transforms certain modes of PIL Images into [0, 1]-ranged tensors with ToTensor(),
    #   MinMaxNormalize() is kinda obsolete here. Nonetheless, I now have the knowledge of custom transformation
    #   implementation because of it, and will keep its call here for didatic purposes.
    #
    transformations = Compose([
        ToTensor(),
        MinMaxNormalize(),
        Flatten()           # Useless: nn.Flatten() exists
    ])

    mnist = MNIST(root='datasets/', train=True, download=True, transform=transformations)

    construction_size = int( 0.8 * len(mnist) )
    test_size         = len(mnist) - construction_size
    construction_raw, test_raw = random_split(mnist, [construction_size, test_size])
    
    print(construction_raw[0][0])
    print(construction_raw[0][0].max(), construction_raw[0][0].min())


def trainae():
    print("It's simpleae-ing time")

    trnsf = Compose([
        ToTensor(),
        MinMaxNormalize()
    ])

    mnist = MNIST(root='datasets/', train=True, download=True, transform=trnsf)

    train_size = int( 0.64 * len(mnist) )
    val_size   = int( 0.16 * len(mnist) )
    test_size  = len(mnist) - (train_size + val_size)
    train, val, test = random_split(mnist, [train_size, val_size, test_size])

    # Defining DataLoader for model
    train_DL = DataLoader(train, batch_size=32, shuffle=True)
    val_DL   = DataLoader(val, batch_size=32, shuffle=False)
    test_DL  = DataLoader(test, batch_size=32, shuffle=False)

    # Using model
    model = SimpleAutoencoder()
    trainer = pl.Trainer(limit_train_batches=1000, max_epochs=10)
    trainer.fit(model, train_dataloaders=train_DL, val_dataloaders=val_DL)



def learn_run(args):
    
    if args.lintrans:
        linearTransformation()

    if args.mnist:  # Transform a version of this into a class
        mnist_load()

    if args.split:
        split()

    if args.prep:
        preprocess()

    if args.simpleae:
        trainae()




