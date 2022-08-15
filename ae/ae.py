from utils import _setup_parser
from run   import learn_run


# Quero três funcionalidades principais:
#   - Uma classe com o modelo de aprendizado
#   - Uma classe para trazer o dataset MNIST
#   - Uma classe para normalizar (preprocessar) os dados



if __name__ == '__main__':

    args = _setup_parser()
    learn_run(args)

