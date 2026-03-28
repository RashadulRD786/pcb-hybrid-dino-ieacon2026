# Paper-Ready Summary

## Experiment

Full-dataset evaluation on the original PCB images using golden-reference guidance.
Methods: pixel differencing, frozen DINOv2 feature differencing, and a board-held-out hybrid fusion.

## Overall Results

- `pixel_diff`: mean AP=0.5313, median AP=0.5201, mean AUC=0.7865, top1=0.9365, top5=0.9942
- `dinov2_diff`: mean AP=0.4032, median AP=0.4061, mean AUC=0.9198, top1=0.6926, top5=0.9163
- `hybrid_pixel_dino`: mean AP=0.6092, median AP=0.6179, mean AUC=0.8890, top1=0.9365, top5=0.9942

## Hybrid Selection

Board-held-out alpha selection used a grid over 21 values. Selected alpha across folds: mean=0.755, min=0.750, max=0.800.

## Best Classes For Hybrid

- `Open_circuit`: hybrid mean AP=0.7042
- `Mouse_bite`: hybrid mean AP=0.6258
- `Spurious_copper`: hybrid mean AP=0.6209

## Practical Reading

- Pixel differencing remains a very strong classical baseline on aligned PCB pairs.
- DINOv2 alone is weaker in mean AP, but adds complementary signal.
- The hybrid fusion improves the overall mean AP and mean AUC over pixel differencing alone.
- This supports a short conference paper centered on reference-guided hybrid localization rather than a large new detector.
