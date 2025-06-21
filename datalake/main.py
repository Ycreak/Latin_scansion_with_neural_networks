from arg_parser import parse_arguments 
from sources.hypotactic import orchestrator as hypotactic_orchestrator
from sources.pedecerto import orchestrator as pedecerto_orchestrator
from models.lstm import orchestrator as lstm_orchestrator
from clean import clean

from datasets import dataset_creator

def main() -> None:
    args: dict = parse_arguments()

    if args.hypotactic:
        print(f"Hypotactic arguments: {args.hypotactic}")
        hypotactic_orchestrator.run(args.hypotactic.split(','))
    
    if args.pedecerto:
        print(f"Pedecerto arguments: {args.pedecerto}")
        pedecerto_orchestrator.run(args.pedecerto.split(','))
    
    if args.clean:
        print(f"Clean arguments: {args.clean}")
        clean.run(args.clean.split(','))
    
    if args.dataset:
        print(f"Dataset arguments: {args.dataset}")
        dataset_creator.run(args.dataset.split(','))

    if args.lstm:
        print(f"LSTM arguments: {args.lstm}")
        lstm_orchestrator.run(args.lstm.split(','))

if __name__ == '__main__':
    main()
