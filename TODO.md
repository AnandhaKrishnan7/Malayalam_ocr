# OCR Improvement TODO

## Current Work
Improving handwritten Malayalam OCR accuracy in app/app.py.

## Steps:
- [ ] 1. Enhance preprocess_for_handwriting: Add adaptive thresholding, deskewing, refined kernels.
- [ ] 2. Update OCR: Try multiple Tesseract configs (OEM1 LSTM, PSM 7/8/13), select best by confidence/length.
- [ ] 3. Add debug logging for extracted texts.
- [ ] 4. Test with existing uploads (e.g., run app, upload 6.jpg).
- [ ] 5. Verify improvements.

Key changes:
- Deskew using minAreaRect.
- Adaptive thresh cv2.ADAPTIVE_THRESH_GAUSSIAN_C.
- OCR voting for robustness.
