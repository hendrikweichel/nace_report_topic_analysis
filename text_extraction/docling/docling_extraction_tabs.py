from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter
#from docling.models.layout import LayoutModelConfig  # path may differ
import tqdm
import os
import glob

pdfs_path = "../../data/PDF_stoxx600"
texts_path = "../../data/TEXT_stoxx600_docling"
pdfs = glob.glob(pdfs_path + "/*.pdf")
print(len(pdfs))

already_done = glob.glob("../../data/TEXT_stoxx600_docling/*.txt")
already_done = [os.path.basename(i)[:-4] for i in already_done]
pdfs = [pdf for pdf in pdfs if os.path.basename(pdf)[:-4] not in already_done]
print(len(pdfs))

pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.mode = TableFormerMode.FAST

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
for pdf in tqdm.tqdm(pdfs):
    result = doc_converter.convert(pdf)
    with open(os.path.join(texts_path, os.path.basename(pdf)[:-4])+".txt", "w") as f: 
        f.write(result.document.export_to_markdown())
