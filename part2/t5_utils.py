import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

T5_CHECKPOINT = 'google-t5/t5-small'

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    wandb.init(project='dsga1011-a4-t5', name=args.experiment_name, config=vars(args))

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(T5_CHECKPOINT)
    else:
        # Same architecture, randomly initialised weights (extra credit)
        config = T5Config.from_pretrained(T5_CHECKPOINT)
        # With the word-level SQL tokenizer the shared embedding / output layer gets one extra row
        # per SQL word (32128 -> 32772); everything else is the unchanged t5-small architecture.
        from load_data import target_vocab_size
        config.vocab_size = target_vocab_size()
        model = T5ForConditionalGeneration(config)

    freeze = getattr(args, 'freeze', 'none')
    if freeze == 'embeddings':
        # shared token embeddings (also tied to the LM head in t5-small)
        model.shared.weight.requires_grad = False
    elif freeze == 'encoder':
        for p in model.encoder.parameters():
            p.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {'pretrained ' + T5_CHECKPOINT if args.finetune else 'T5-small from scratch'} | "
          f"vocab {model.config.vocab_size} | "
          f"freeze={freeze} | trainable params {n_train:,} / {n_total:,}")
    return model.to(DEVICE)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def get_checkpoint_dir(args):
    # Must match the path built in train_t5.train()
    model_type = 'ft' if args.finetune else 'scr'
    return os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    # checkpoints/<ft|scr>_experiments/<name>/{best,last}/ (config.json + model.safetensors)
    path = os.path.join(checkpoint_dir, 'best' if best else 'last')
    mkdir(path)
    model.save_pretrained(path)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    path = os.path.join(get_checkpoint_dir(args), 'best' if best else 'last')
    model = T5ForConditionalGeneration.from_pretrained(path)
    return model.to(DEVICE)

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

