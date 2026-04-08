# Utils for slide data ingest and organization

# Organization
- `annot_fsx`: Distingush frozen vs permanent of frozen images
    - `frozen_slide_classifier.ipynb`: I believe `frozen_slide_classifier` is
      an early attempt to classify based on `i` and `+` markings. It is not
      reliable and should be avoided
    - `save_thumbnail_embedding.py`: We save embeddings of thumbnail images to
      help us order the images during annotation
    - `save_annot_batch_im_csv.ipynb`: We save thumbnail images using embedding
      order and annotate frozen vs permanent of frozen images
- `mask_recog`: Recognize whats on the masking tape used during MLiNS scanning.
  This is for the earlier phase where Renly, Samir, Asadur, and Cheng scanned
  at NCRC
    - `run_handwritten_ocr.ipynb`: run handwritten OCR using LLAVA, only used
      for the first weekend of scanning, not very reliable
    - `review_handwritten_ocr.ipynb`: review the handwritten OCR
    - `read_qrcode.ipynb`: The bulk of data is processed this way, it reads the
      qrcode printed on the label, but it can miss occationally too.

# Scanning phases:
- Phase A: MLiNS scan, first batch of SRH matched, ~1518 SUs.
- Phase B: MLiNS scan, SUs missed from Phase A, mostly cases from brain tumor
  board. ~758 SUs were requested. Phases A and B are processed the same way,
  except for first weekend of Phase A.
- Phase C: DP staff scan, remaining NIO paired data, 458 SUs were requested.
