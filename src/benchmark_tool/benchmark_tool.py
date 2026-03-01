from src.benchmark_tool.argparser import parse_args
from src.model_wrappers.full_tabpfn_gen import FullTabpfnGen
from src.model_wrappers.smote_generator import SmoteGenerator
from src.model_wrappers.ctgan_generator import CTGANGenerator


def get_model_class(args):
    match args.generator_type.lower():
        case "tabiclgen":
            pass
        case "smote":
            return SmoteGenerator
        case "ctgan":
            return CTGANGenerator
        case "tabpfnunsupervised":
            return FullTabpfnGen
        case _:
            raise Exception("Chosen generator type is incorrect")


def main():
    args = parse_args()
    generator_model_class = get_model_class(args)
    print(generator_model_class)
