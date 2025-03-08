from library.SDXLProcessor import SDXLImageProcessor

processor = SDXLImageProcessor(
    gpu_id=1,
    csv_path="parte_5.csv",
)

processor.process_images_batch()
