"""
initialize_model.py
===================

"""

import torch
from pionono_models.model_supervised import SupervisedSegmentationModel
from pionono_models.model_confusionmatrix import ConfusionMatrixModel
from models.dpersona import DPersona
from Probabilistic_Unet_Pytorch.probabilistic_unet import ProbabilisticUnet
from pionono_models.model_pionono import PiononoModel

# ============================================================
# ============================================================
ORIGINAL_BACKBONE_DEFAULT = False

def init_model(args, opt):

    if args.model_name == 'prob_unet':
        model = ProbabilisticUnet(
            input_channels=opt.INPUT_CHANNEL,
            num_classes=opt.OUTPUT_CHANNEL,
            latent_dim=6, no_convs_fcomb=4,
            alpha=1.0, reg_factor=0.00001,
            original_backbone=ORIGINAL_BACKBONE_DEFAULT
        )

    elif args.model_name == 'MOSAIC':
        from models.mosaic import MOSAIC
        
        model = MOSAIC(
            input_channels=opt.INPUT_CHANNEL,
            num_classes=opt.OUTPUT_CHANNEL,
            latent_dim=6, no_convs_fcomb=4,
            num_experts=args.mask_num,
            reg_factor=0.00001,
            original_backbone=ORIGINAL_BACKBONE_DEFAULT,
            num_annotators=args.mask_num,
            num_timesteps=getattr(args, 'num_timesteps', 200),
            inference_steps=getattr(args, 'inference_steps', 5),
            diff_loss_weight=getattr(args, 'diff_loss_weight', 0.5),
            prediction_type=getattr(args, 'prediction_type', 'x0'),
            inference_t0_ratio=getattr(args, 'inference_t0_ratio', 0.1),
            style_bank_dim=getattr(args, 'style_bank_dim', 32),
            boundary_refine_ch=getattr(args, 'boundary_refine_ch', 16),
            sabr_sigma=getattr(args, 'sabr_sigma', 2.0),
            bdry_warmup_epochs=getattr(args, 'bdry_warmup_epochs', 40),
            ecrd_train_steps=getattr(args, 'ecrd_train_steps', 50),
            ecrd_inference_steps=getattr(args, 'ecrd_inference_steps', 5),
            ecrd_hidden_ch=32,
            ecrd_ready_threshold=getattr(args, 'ecrd_ready_threshold', 0.08),
            early_stopping_patience=getattr(args, 'early_stop_patience', 30),
        )

        backbone_str = "double-conv U-Net (Probabilistic_Unet_Pytorch.unet.Unet)" \
                       if ORIGINAL_BACKBONE_DEFAULT else "SMP ResNet34 (UnetHeadless)"
        print("[init_model] MOSAIC: "
              "experts={}, annotators={}, style_dim={}".format(
                  args.mask_num, args.mask_num, model.style_bank_dim))
        print("  [BACKBONE] original_backbone={} -> {}".format(
            ORIGINAL_BACKBONE_DEFAULT, backbone_str))
        print("  SABR: sigma={}, bdry_warmup={} epochs, refine_ch={}".format(
            model.spatial_boundary_detector.sigma,
            model.bdry_warmup_epochs,
            getattr(args, 'boundary_refine_ch', 16)))
        print("  SC-ECRD: train_steps={}, infer_steps={}, ready_threshold={}".format(
            model.ecrd_train_steps, model.ecrd_inference_steps,
            model.ecrd_start_threshold))
        print("  EarlyStopping: patience={}".format(model.early_stopping.patience))

        module_names = [
            ('style_bank', 'Style Bank'),
            ('boundary_refiner', 'Boundary Refiner (SABR)'),
            ('spatial_boundary_detector', 'SABR Detector (buffers)'),
            ('diff_z_prior', 'ECRD Diffusion Prior'),
            ('fcomb', 'EvidentialFcomb'),
            ('unet', 'U-Net Backbone'),
        ]
        total_params = sum(p.numel() for p in model.parameters())
        for attr_name, display_name in module_names:
            if hasattr(model, attr_name):
                mod = getattr(model, attr_name)
                n_learn = sum(p.numel() for p in mod.parameters())
                n_buf = sum(b.numel() for b in mod.buffers())
                if n_learn > 0:
                    print("  {}: {:,} params ({:.1f}%)".format(
                        display_name, n_learn, n_learn/total_params*100))
                elif n_buf > 0:
                    print("  {}: {:,} buffer elements (no learnable)".format(
                        display_name, n_buf))

    elif 'DPersona' in args.model_name:
        model = DPersona(
            input_channels=opt.INPUT_CHANNEL,
            num_classes=opt.OUTPUT_CHANNEL,
            latent_dim=6, no_convs_fcomb=4,
            num_experts=args.mask_num,
            reg_factor=0.00001,
            original_backbone=ORIGINAL_BACKBONE_DEFAULT
        )

    elif args.model_name == 'pionono':
        annotator_list = list(range(args.mask_num))
        model = PiononoModel(
            input_channels=opt.INPUT_CHANNEL,
            num_classes=opt.OUTPUT_CHANNEL,
            annotators=annotator_list, gold_annotators=0,
            latent_dim=8, no_head_layers=3,
            head_kernelsize=1, head_dilation=1,
            kl_factor=0.0005, reg_factor=0.00001,
            mc_samples=5, z_prior_sigma=2.0,
            z_posterior_init_sigma=8.0,
        )

    elif args.model_name == 'cm_global':
        model = ConfusionMatrixModel(
            input_channels=opt.INPUT_CHANNEL,
            num_classes=opt.OUTPUT_CHANNEL,
            num_annotators=args.mask_num,
            level='global', image_res=opt.INPUT_SIZE,
            learning_rate=0.001, alpha=1.0, min_trace=False
        )

    elif args.model_name == 'cm_pixel':
        model = ConfusionMatrixModel(
            input_channels=opt.INPUT_CHANNEL,
            num_classes=opt.OUTPUT_CHANNEL,
            num_annotators=args.mask_num,
            level='pixel', image_res=opt.INPUT_SIZE,
            learning_rate=0.001, alpha=1.0, min_trace=False
        )

    else:
        model = SupervisedSegmentationModel(opt.INPUT_CHANNEL)

    return model