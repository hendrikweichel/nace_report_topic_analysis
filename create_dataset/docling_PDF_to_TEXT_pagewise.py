from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
import tqdm
import os
import glob
import json

pdfs_path = "projects/nace_classification/nace_report_topic_analysis/data/datasets/reports_subset_from_full_data_2/PDFs"
texts_path = "projects/nace_classification/nace_report_topic_analysis/data/datasets/reports_subset_from_full_data_2/TXTs"

overwrite = False

pdfs = glob.glob(pdfs_path + "/*.pdf")

pipeline_options = PdfPipelineOptions(
    # your current options
)
# choose GPU device
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=8, 
    device=AcceleratorDevice.CUDA 
)
pipeline_options = PdfPipelineOptions(do_table_structure=True)
#pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# assume you already built pipeline_options above
doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

os.makedirs(texts_path, exist_ok=True)

print(f"Converting {len(pdfs)} PDFs in {pdfs_path} to TXT in file {texts_path}!")

for pdf in tqdm.tqdm(pdfs):
    #try:
    if True: 
        print("Converting:", pdf)

        out_path = os.path.join(
            texts_path, os.path.splitext(os.path.basename(pdf))[0] + ".txt"
        )        
       
        if not overwrite: 
            if os.path.isfile(out_path): 
                continue

        try: 
            result = doc_converter.convert(pdf)
        except RuntimeError:
            continue
        doc = result.document
        
        pages = ""
        for i in range(doc.num_pages()):              # 0-based
            md = doc.export_to_markdown(page_no=i)  # page-wise export
            pages += md 
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pages)
            print("stored at", out_path)

