"""Add vector print pages proving the native figure-plus-caption height.

Reads only owned companion PDFs and preserved captions. Original files and
render manifest remain unchanged. No scientific source or model reads.
"""
from pathlib import Path
import argparse
import hashlib
import json
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties


def wrap_caption(value):
    # Measure at the exact caption validation canvas DPI. Font hinting at 72
    # versus 100 DPI can move a line beyond the right edge by several pixels.
    renderer = RendererAgg(670, 1, 100)
    font = FontProperties(family='DejaVu Sans', size=10)
    lines, current = [], ''
    for word in value.split():
        candidate = (current + ' ' + word).strip()
        if renderer.get_text_width_height_descent(candidate, font, False)[0] > 669.5 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    captions = json.loads((root / 'captions.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'savefig.bbox': None})
    output = []
    scratch = root / 'print_pages_v4'
    assert not scratch.exists()
    scratch.mkdir()
    for name, caption in captions.items():
        available = psutil.virtual_memory().available
        assert available >= 1.2 * 1024 ** 3, 'Defer caption rendering below 1.2 GiB available RAM'
        lines = wrap_caption(caption)
        assert ' '.join(lines) == ' '.join(caption.split())
        caption_height_pt = len(lines) * 12
        figure = plt.figure(figsize=(6.7, caption_height_pt / 72), dpi=100)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.set(xlim=(0, 6.7 * 72), ylim=(0, caption_height_pt))
        axes.axis('off')
        labels = [axes.text(0, 3 + 12 * (len(lines) - 1 - i), line, fontsize=10,
                            ha='left', va='baseline', parse_math=False) for i, line in enumerate(lines)]
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        width_px, height_px = figure.canvas.get_width_height()
        for label in labels:
            bbox = label.get_window_extent(renderer)
            assert bbox.x0 >= -0.1 and bbox.x1 <= width_px + 0.1 and bbox.y0 >= -0.1 and bbox.y1 <= height_px + 0.1
        caption_pdf = scratch / (name + '_caption.pdf')
        figure.savefig(caption_pdf, bbox_inches=None, pad_inches=0)
        plt.close(figure)
        caption_reader = PdfReader(caption_pdf)
        caption_text = caption_reader.pages[0].extract_text()
        # PDF text extraction can insert a space within a kerned pair, e.g.
        # "Target" -> "T arget". Preserve exact words in the drawn labels and
        # additionally check every non-whitespace PDF glyph in order.
        assert ''.join(caption_text.split()) == ''.join(caption.split())
        original_reader = PdfReader(root / (name + '.pdf'))
        original_page = original_reader.pages[0]
        width_pt = float(original_page.mediabox.width)
        figure_height_pt = float(original_page.mediabox.height)
        assert abs(width_pt - 482.4) < 1e-8
        gap_pt = 0.15 * 72
        total_height_pt = figure_height_pt + caption_height_pt + gap_pt
        assert total_height_pt / 72 <= 8.3
        writer = PdfWriter()
        page = writer.add_blank_page(width=width_pt, height=total_height_pt)
        page.merge_transformed_page(original_page, Transformation().translate(ty=caption_height_pt + gap_pt))
        page.merge_page(caption_reader.pages[0])
        destination = scratch / (name + '_with_caption.pdf')
        assert not destination.exists()
        with destination.open('wb') as stream:
            writer.write(stream)
        check = PdfReader(destination).pages[0]
        fonts = []
        check.extract_text(visitor_text=lambda text, cm, tm, font, size: fonts.append(float(size)) if text.strip() else None)
        assert fonts and min(fonts) >= 10
        assert abs(float(check.mediabox.width) - 482.4) < 1e-8
        assert abs(float(check.mediabox.height) - total_height_pt) < 1e-8
        assert ''.join(caption.split()) in ''.join(check.extract_text().split())
        output.append({'figure': name, 'path': destination.relative_to(root).as_posix(), 'sha256': sha(destination),
                       'page_width_in': width_pt / 72, 'page_height_in': total_height_pt / 72,
                       'minimum_pdf_text_font_pt': min(fonts), 'caption_exact_word_sequence': True,
                       'pdf_caption_nonwhitespace_glyph_sequence_identical': True,
                       'caption_line_count': len(lines), 'system_available_memory_before_render_bytes': available})
    receipt = {'status': 'PASS_NATIVE_VECTOR_PRINT_PAGES',
               'parent_render_manifest_sha256': sha(root / 'manifest.json'),
               'caption_files_unchanged': {name: sha(root / name) for name in ['captions.json', 'captions.txt']},
               'generator_sha256': sha(__file__), 'files': output,
               'scope': 'Owned vector companion diagrams plus preserved exact captions; 10pt caption text, 12pt baselines, 0.15in separation.',
               'no_original_or_scientific_input_edits': True, 'scientific_results_read': False}
    (root / 'captioned_pages_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'pages': output}, indent=2))


if __name__ == '__main__':
    main()
