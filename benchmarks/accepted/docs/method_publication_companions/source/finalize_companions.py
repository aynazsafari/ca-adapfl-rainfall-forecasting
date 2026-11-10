"""Assemble only passed companion assets; never read model results or arrays."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import re
import xml.etree.ElementTree as ET
from PIL import Image
from pypdf import PdfReader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate, original, repo, out = [p.resolve() for p in [args.candidate, args.original, args.repo, args.output]]
    assert not out.exists(), 'Use a fresh final directory'
    render = json.loads((candidate / 'manifest.json').read_text())
    original_manifest = json.loads((original / 'manifest.json').read_text())
    print_pages = json.loads((candidate / 'captioned_pages_receipt.json').read_text())
    assert sha(candidate / 'manifest.json') == print_pages['parent_render_manifest_sha256']
    assert sha(original / 'manifest.json') == render['original_manifest_sha256']
    preserved = []
    for entry in original_manifest['files']:
        assert sha(original / entry['path']) == entry['sha256']
        assert sha(repo / 'docs/method_figures' / entry['path']) == entry['sha256']
        preserved.append({'path': entry['path'], 'sha256': entry['sha256'], 'original_and_repo_copy_unchanged': True})
    for key, path in render['source_paths_read_or_hashed'].items():
        assert sha(path) == render['source_sha256'][key]
    for name, expected in render['source_sha256']['scientific_modules'].items():
        assert sha(repo / 'caadapfl' / (name + '.py')) == expected
    checked = []
    for name, layout in render['layout'].items():
        with Image.open(candidate / (name + '.png')) as img:
            assert img.width == 2010 and img.height == round(layout['native_height_in'] * 300)
            assert all(abs(dpi - 300) < 0.001 for dpi in img.info['dpi'])
            pixels = [img.width, img.height]
        svg = ET.parse(candidate / (name + '.svg')).getroot()
        assert abs(float(svg.attrib['width'].rstrip('pt')) - 482.4) < 1e-8
        spans = [float(x) for el in svg.iter() for x in re.findall(r'font-size:\s*([\d.]+)px', el.attrib.get('style', ''))]
        assert spans and min(spans) >= 10
        pdf = PdfReader(candidate / (name + '.pdf')).pages[0]
        fonts = []
        pdf.extract_text(visitor_text=lambda text, cm, tm, font, size: fonts.append(float(size)) if text.strip() else None)
        assert fonts and min(fonts) >= 10
        page = next(p for p in print_pages['files'] if p['figure'] == name)
        assert sha(candidate / page['path']) == page['sha256']
        assert page['page_height_in'] <= 8.3 and page['minimum_pdf_text_font_pt'] >= 10
        checked.append({'figure': name, 'native_width_in': 6.7, 'diagram_height_in': layout['native_height_in'],
                        'png_pixels': pixels, 'png_dpi': 300, 'minimum_svg_and_pdf_label_font_pt': min(min(spans), min(fonts)),
                        'captioned_pdf': Path(page['path']).name, 'captioned_pdf_sha256': page['sha256'],
                        'captioned_pdf_height_in': page['page_height_in'], 'caption_font_pt': 10,
                        'caption_exact_original_words': page['caption_exact_word_sequence'],
                        'caption_pdf_glyph_sequence_check': page['pdf_caption_nonwhitespace_glyph_sequence_identical'],
                        'original_resolution_visual_review': 'PASS_BY_AUTHOR_AND_INDEPENDENT_AGENT'})
    for name in ['captions.json', 'captions.txt']:
        assert sha(candidate / name) == sha(original / name)
    out.mkdir()
    for entry in render['files']:
        assert sha(candidate / entry['path']) == entry['sha256']
        shutil.copy2(candidate / entry['path'], out / entry['path'])
    for page in print_pages['files']:
        shutil.copy2(candidate / page['path'], out / Path(page['path']).name)
    shutil.copy2(candidate / 'manifest.json', out / 'render_manifest.json')
    shutil.copy2(candidate / 'captioned_pages_receipt.json', out / 'captioned_pages_source_receipt.json')
    print_receipt = dict(print_pages)
    print_receipt['source_receipt_sha256'] = sha(candidate / 'captioned_pages_receipt.json')
    print_receipt['parent_render_manifest_path'] = 'render_manifest.json'
    print_receipt['files'] = [dict(page, path=Path(page['path']).name) for page in print_pages['files']]
    save(out / 'captioned_pages_receipt.json', print_receipt)
    shutil.copy2(original / 'manifest.json', out / 'original_v3_manifest.json')
    shutil.copy2(Path(__file__).parent / 'font_assessment.json', out / 'font_assessment.json')
    (out / 'source').mkdir()
    for name in ['make_companions.py', 'add_captioned_pdfs.py', 'finalize_companions.py']:
        shutil.copy2(Path(__file__).parent / name, out / 'source' / name)
    independent = {
        'status': 'PASS', 'reviewer': '/root/audit_notebook', 'evidence_type': 'Direct agent review message; recorded by assembling agent',
        'render_manifest_sha256': sha(candidate / 'manifest.json'),
        'scope': 'All four companion PNGs individually at original resolution, all four original PNGs, captions, and both generators.',
        'findings': ['No clipping, overlap, arrow, dimension, unit, support-calendar or domain defect observed.',
                     'Every SVG minimum font 10.5pt; native PNG widths 2010 pixels at 300dpi.',
                     'Network nonlinear attention, bypass and two heads; two-line objective preserved.',
                     'Plain summaries, reporting 16bytes/client, selected inactive aggregation terms versus anchor, privileged graph/raw pooling/noDP explicit.',
                     'Lag support and vintage assumptions match originals; frozen boxes and C1-C5 overlap with native-center/WGS84 whole-cell weighting retained.'],
        'scientific_results_or_models_read': False, 'reviewer_writes_or_renders': False}
    save(out / 'independent_visual_review.json', independent)
    (out / 'README.txt').write_text(
        'Method figure publication companions\n\n'
        'Insert each standalone figure at its native width of 6.7 inches. Diagram labels are at least 10.5pt; '
        'PNG files are 2010 pixels wide at 300dpi. PDF and SVG files preserve vector text and shapes. '
        'The four *_with_caption.pdf pages combine the vector diagrams with the exact original captions '
        'in 10pt type, 12pt baselines and a 0.15-inch gap. Their heights are 8.150, 8.017, 7.550 and 7.617 inches.\n\n'
        'captions.json and captions.txt are byte-identical to the frozen v3 captions. captions_print_10pt.txt '
        'is an optional line-wrap example; final captioned PDFs use DPI-matched wrapping. PDF extraction may '
        'insert whitespace inside kerned letter pairs, so PDF glyph order is checked ignoring whitespace; '
        'the drawn caption labels retain the exact original word sequence.\n\n'
        'Both author and independent reviewer inspected all four PNGs at original resolution. Captioned PDF '
        'page geometry, text fonts, caption label bounds and glyph sequence were checked mechanically; '
        'no separate full-page PDF raster inspection is claimed. These are source-only layout derivatives. '
        'No accuracy, prediction, evaluation-truth, checkpoint or round-history data were opened. Frozen '
        'v3 originals, repo copies and scientific modules remain unchanged.\n', encoding='utf-8')
    assets = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
              for p in sorted(out.rglob('*')) if p.is_file()]
    manifest = {'status': 'PASS_SOURCE_ONLY_PUBLICATION_COMPANIONS', 'scope': 'Four native 6.7-inch method diagrams and original captions; no result figure or scientific change.',
                'render_manifest_sha256': sha(candidate / 'manifest.json'), 'original_manifest_sha256': sha(original / 'manifest.json'),
                'source_sha256': render['source_sha256'], 'font_assessment_sha256': render['font_assessment_sha256'],
                'scientific_results_or_models_read': False, 'original_and_repository_files_preserved': preserved,
                'figures': checked, 'independent_review_sha256': sha(out / 'independent_visual_review.json'),
                'limitations': ['10.5pt labels apply at native 6.7-inch width; smaller insertion widths scale text.',
                                'Captioned PDF page geometry/text checked mechanically; original-resolution visual inspection used standalone PNGs.'],
                'files': assets}
    save(out / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'directory': str(out), 'manifest_sha256': sha(out / 'manifest.json'), 'figures': checked}, indent=2))


if __name__ == '__main__':
    main()
