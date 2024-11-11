import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
import torch.quantization
import logging

logger = logging.getLogger(__name__)

@DETECTORS.register_module()
class LightestECLRNet(SingleStageDetector):
    """Lightest ECLRNet detector for lane detection with production optimizations"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(LightestECLRNet, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg
        )

        # Initialize quantization components
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Model optimization configs
        self.temperature = 2.0  # Knowledge distillation temperature
        self.max_batch_size = 32
        self.enable_fusion = True
        self.enable_cache = True

        # Feature cache
        self._cached_features = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def prepare_for_inference(self):
        """Optimize model for inference"""
        try:
            self.eval()
            if self.enable_fusion:
                # Fuse conv + bn + relu modules
                self._fuse_modules()

            # Configure quantization
            self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self, inplace=True)

            logger.info("Model prepared for inference with optimizations")

        except Exception as e:
            logger.error(f"Failed to prepare for inference: {str(e)}")
            raise

    def _fuse_modules(self):
        """Fuse conv + bn + relu modules for inference optimization"""
        modules_to_fuse = []
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                modules_to_fuse.append(name)

        if modules_to_fuse:
            torch.quantization.fuse_modules(self, modules_to_fuse, inplace=True)
            logger.info(f"Fused {len(modules_to_fuse)} modules")

    def extract_feat(self, img):
        """Extract features with optimizations"""
        try:
            if not isinstance(img, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor")

            # Apply quantization in inference mode
            if not self.training and self.quant is not None:
                img = self.quant(img)

            # Check cache if enabled
            if self.enable_cache and not self.training:
                cache_key = hash(img.data_ptr())
                if cache_key in self._cached_features:
                    self._cache_hits += 1
                    return self._cached_features[cache_key]
                self._cache_misses += 1

            # Extract features
            x = self.backbone(img)

            if isinstance(x, (list, tuple)):
                if not all(isinstance(feat, torch.Tensor) for feat in x):
                    raise TypeError("All features must be torch.Tensor")

            if self.with_neck:
                x = self.neck(x)

            # Dequantize if needed
            if not self.training and self.dequant is not None:
                x = [self.dequant(feat) for feat in x]

            # Cache features if enabled
            if self.enable_cache and not self.training:
                self._cached_features[cache_key] = x

                # Limit cache size
                if len(self._cached_features) > 1000:
                    self._cached_features.clear()

            return x

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def get_inference_stats(self):
        """Get inference optimization statistics"""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }