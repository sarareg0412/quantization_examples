from collections import OrderedDict


class Mapping:

    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
        [
            # Model for Image Classification mapping
            ("beit", "BeitForImageClassification"),
            ("bit", "BitForImageClassification"),
            ("convnext", "ConvNextForImageClassification"),
            ("convnextv2", "ConvNextV2ForImageClassification"),
            ("cvt", "CvtForImageClassification"),
            ("data2vec-vision", "Data2VecVisionForImageClassification"),
            ("deit", ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher")),
            ("dinat", "DinatForImageClassification"),
            ("dinov2", "Dinov2ForImageClassification"),
            (
                "efficientformer",
                (
                    "EfficientFormerForImageClassification",
                    "EfficientFormerForImageClassificationWithTeacher",
                ),
            ),
            ("efficientnet", "EfficientNetForImageClassification"),
            ("focalnet", "FocalNetForImageClassification"),
            ("imagegpt", "ImageGPTForImageClassification"),
            ("levit", ("LevitForImageClassification", "LevitForImageClassificationWithTeacher")),
            ("mobilenet_v1", "MobileNetV1ForImageClassification"),
            ("mobilenet_v2", "MobileNetV2ForImageClassification"),
            ("mobilevit", "MobileViTForImageClassification"),
            ("mobilevitv2", "MobileViTV2ForImageClassification"),
            ("nat", "NatForImageClassification"),
            (
                "perceiver",
                (
                    "PerceiverForImageClassificationLearned",
                    "PerceiverForImageClassificationFourier",
                    "PerceiverForImageClassificationConvProcessing",
                ),
            ),
            ("poolformer", "PoolFormerForImageClassification"),
            ("pvt", "PvtForImageClassification"),
            ("regnet", "RegNetForImageClassification"),
            ("resnet", "ResNetForImageClassification"),
            ("segformer", "SegformerForImageClassification"),
            ("swiftformer", "SwiftFormerForImageClassification"),
            ("swin", "SwinForImageClassification"),
            ("swinv2", "Swinv2ForImageClassification"),
            ("van", "VanForImageClassification"),
            ("vit", "ViTForImageClassification"),
            ("vit_hybrid", "ViTHybridForImageClassification"),
            ("vit_msn", "ViTMSNForImageClassification"),
        ]
    )

    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
        [
            # Model for Semantic Segmentation mapping
            ("beit", "BeitForSemanticSegmentation"),
            ("data2vec-vision", "Data2VecVisionForSemanticSegmentation"),
            ("dpt", "DPTForSemanticSegmentation"),
            ("mobilenet_v2", "MobileNetV2ForSemanticSegmentation"),
            ("mobilevit", "MobileViTForSemanticSegmentation"),
            ("mobilevitv2", "MobileViTV2ForSemanticSegmentation"),
            ("segformer", "SegformerForSemanticSegmentation"),
            ("upernet", "UperNetForSemanticSegmentation"),
        ]
    )
    MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict(
        [
            # Model for Universal Segmentation mapping
            ("detr", "DetrForSegmentation"),
            ("mask2former", "Mask2FormerForUniversalSegmentation"),
            ("maskformer", "MaskFormerForInstanceSegmentation"),
            ("oneformer", "OneFormerForUniversalSegmentation"),
        ]
    )
    # Automatic speech recognition
    MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
        [
            # Model for Connectionist temporal classification (CTC) mapping
            ("data2vec-audio", "Data2VecAudioForCTC"),
            ("hubert", "HubertForCTC"),
            ("mctct", "MCTCTForCTC"),
            ("sew", "SEWForCTC"),
            ("sew-d", "SEWDForCTC"),
            ("unispeech", "UniSpeechForCTC"),
            ("unispeech-sat", "UniSpeechSatForCTC"),
            ("wav2vec2", "Wav2Vec2ForCTC"),
            ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC"),
            ("wavlm", "WavLMForCTC"),
        ]
    )

    #text2text - generation -> AutoModelForSeq2SeqLM
