from library.SDXLProcessor import SDXLImageProcessor

processor = SDXLImageProcessor(
    gpu_id=1,
    csv_path="parte_4.csv",
)

processor.process_images_batch()
