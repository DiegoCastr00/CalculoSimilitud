from library.SDXLProcessor import SDXLImageProcessor

processor = SDXLImageProcessor(
    gpu_id=0,
    csv_path="parte_1.csv",
)

processor.process_images_batch()
