import argparse


def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--subject",  type=int, required=True) 
    parser.add_argument("--hidden_layer_dim",  type=int, required=True)
    parser.add_argument("--hidden_layer_num",  type=int, required=True)
    parser.add_argument("--lags",  nargs="+", type=int, required=True)
    parser.add_argument("--EPOCHS", type=int, required=True)
    parser.add_argument("--train_num", type=float, required=True) 
    parser.add_argument("--min_num_words", type=int, required=True)
    parser.add_argument("--use_second_network", action="store_true")
    parser.add_argument("--electrode", type=int, required=True)
    parser.add_argument("--taking_words", action="store_true")
    parser.add_argument("--num_words", type=int, required=True)
    parser.add_argument("--activation_function", type=str, required=True)
    parser.add_argument("--learning_rate",  type=float, required=True) 
    parser.add_argument("--momentum",  type=float, required=True) 
    parser.add_argument("--all_elec", action="store_true")
    parser.add_argument("--analysis_level", type=str, required=True)
    parser.add_argument("--max_duration", type=int, required=True)
    parser.add_argument("--HIGH_Value", type=int, default=int(1e5), required=False)
    parser.add_argument("--emb_dim", type=int, default=int(50), required=False)
    parser.add_argument("--save_dir", type=str, default=str('/scratch/gpfs/arnab/flexible_encoding/results/mat_files/'), required=False)
    # parser.add_argument("--selected_elec_id", action="store_true") 
    # parser.add_argument("--across_subject_with_repacing_srm", action="store_true")
    
    # parser.add_argument("--data-dir", type=str, required=True)
    # parser.add_argument("--saving-dir", type=str, required=True)
    # parser.add_argument("--freeze-decoder", action="store_true")
    
    # parser.add_argument("--z_score", action="store_true")
    # parser.add_argument("--make_position_embedding_zero", action="store_true")

    args = parser.parse_args()
    return args