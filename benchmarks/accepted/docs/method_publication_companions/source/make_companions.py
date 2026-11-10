"""Native 6.7-inch method companions; source-only and read-only inputs.

Never opens results, model states, scores, evaluation labels or fit histories.
Preserves frozen v3 files and captions; reflows labels and diagram geometry.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from shapely.geometry import shape, box

WIDTH = 6.7
MIN_FONT = 10.5
MEMORY_GATE = int(1.2 * 1024 ** 3)
COLORS = ['#315B7E', '#B28735', '#71844A', '#A6694A', '#7E6897']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def page(height):
    available = psutil.virtual_memory().available
    require(available >= MEMORY_GATE, 'Render deferred: available RAM is below 1.2 GiB')
    fig = plt.figure(figsize=(WIDTH, height), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, height))
    ax.axis('off')
    return fig, ax, available


def text(ax, x, y, value, *, fs=MIN_FONT, ha='center', va='center', weight='normal', color='#202020'):
    require(fs >= 10, 'Sub-10pt label requested')
    return ax.text(x, y, value, fontsize=fs, ha=ha, va=va, fontweight=weight,
                   color=color, linespacing=1.18, parse_math=False)


def node(ax, x, y, width, height, value, *, fs=MIN_FONT):
    ax.add_patch(Rectangle((x, y), width, height, facecolor='#F1F4F6', edgecolor='#5E6973', linewidth=0.9))
    label = text(ax, x + width / 2, y + height / 2, value, fs=fs)
    label._node_bounds = (x, y, width, height)
    return label


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
                               color='#444B52', linewidth=1.05, shrinkA=1, shrinkB=1))


def wrap_caption(value):
    renderer = RendererAgg(1, 1, 72)
    font = FontProperties(family='DejaVu Sans', size=10)
    lines, current = [], ''
    for word in value.split():
        candidate = (current + ' ' + word).strip()
        width = renderer.get_text_width_height_descent(candidate, font, False)[0]
        if width > WIDTH * 72 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo, original, out = args.repo.resolve(), args.original.resolve(), args.output.resolve()
    require(not out.exists(), 'Use a fresh candidate directory')
    original_manifest = json.loads((original / 'manifest.json').read_text())
    captions = json.loads((original / 'captions.json').read_text(encoding='utf-8'))
    require(len(captions) == 4 and original_manifest['source_only'], 'Frozen figure source set differs')
    original_hashes = {entry['path']: entry['sha256'] for entry in original_manifest['files']}
    for name, expected in original_hashes.items():
        require(sha(original / name) == expected, 'Frozen v3 artifact changed: ' + name)
        require(sha(repo / 'docs/method_figures' / name) == expected, 'Copied repository method artifact differs: ' + name)
    sources = original_manifest['source_sha256']
    run = repo / 'runs/revision_v2'
    source_paths = {'run_identity': run / 'run_identity.json',
                    'schema': run / 'datasets/MAIN_FULL_schema.json',
                    'country_geometry': repo / 'provenance/source_tables/india_country_ne_5_1_1.geojson',
                    'extraction_protocol': repo / 'provenance/source_tables/extraction_protocol.json',
                    'locked_hyperparameters': run / 'development/hyperparameters_frozen.json'}
    for label, path in source_paths.items():
        require(sha(path) == sources[label], 'Frozen diagram source hash differs: ' + label)
    for module, expected in sources['scientific_modules'].items():
        require(sha(repo / 'caadapfl' / (module + '.py')) == expected, 'Scientific module changed')
    protocol = json.loads(source_paths['extraction_protocol'].read_text())
    raw_geometry = json.loads(source_paths['country_geometry'].read_text())
    if raw_geometry.get('type') == 'FeatureCollection':
        country = shape(raw_geometry['features'][0]['geometry'])
    elif raw_geometry.get('type') == 'Feature':
        country = shape(raw_geometry['geometry'])
    else:
        country = shape(raw_geometry)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': MIN_FONT,
                         'axes.titlesize': 13, 'axes.labelsize': MIN_FONT,
                         'xtick.labelsize': MIN_FONT, 'ytick.labelsize': MIN_FONT,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white', 'savefig.bbox': None})
    out.mkdir(parents=True)
    layout = {}
    all_text = {}

    def save(fig, name, available):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [artist for artist in fig.findobj(plt.Text) if artist.get_text() and artist.get_visible()]
        pixel_width, pixel_height = fig.canvas.get_width_height()
        require(all(artist.get_fontsize() >= 10 for artist in labels), 'Sub-10pt text artist')
        for label in labels:
            bounds = label.get_window_extent(renderer)
            require(bounds.x0 >= -0.1 and bounds.x1 <= pixel_width + 0.1 and
                    bounds.y0 >= -0.1 and bounds.y1 <= pixel_height + 0.1,
                    'Label outside native page: ' + label.get_text())
            if hasattr(label, '_node_bounds'):
                x, y, w, h = label._node_bounds
                edge = label.axes.transData.transform([[x, y], [x + w, y + h]])
                require(bounds.x0 >= edge[0, 0] + 2 and bounds.x1 <= edge[1, 0] - 2 and
                        bounds.y0 >= edge[0, 1] + 2 and bounds.y1 <= edge[1, 1] - 2,
                        'Label exceeds node: ' + label.get_text())
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / (name + '.' + suffix), dpi=300, bbox_inches=None, pad_inches=0)
        root = ET.parse(out / (name + '.svg')).getroot()
        require(abs(float(root.get('width')[:-2]) - WIDTH * 72) < 1e-8, 'Native SVG width changed')
        font_sizes = []
        for element in root.iter():
            if element.tag.endswith(('text', 'tspan')):
                font_sizes.extend(float(value) for value in re.findall(r'(?:font-size:\s*|font:[^;]*?)([\d.]+)px', element.get('style', '')))
        require(font_sizes and min(font_sizes) >= 10, 'SVG contains sub-10pt spans')
        caption_lines = wrap_caption(captions[name])
        caption_height = len(caption_lines) * 12 / 72
        native_height = float(fig.get_size_inches()[1])
        require(native_height + caption_height + 0.15 <= 8.3, 'Diagram plus preserved 10pt caption exceeds 8.3 inches')
        all_text[name] = [artist.get_text() for artist in labels]
        layout[name] = {'native_width_in': WIDTH, 'native_height_in': native_height,
                        'minimum_artist_and_svg_font_pt': min(font_sizes),
                        'PNG_dpi': 300, 'caption_exact_original_text_preserved': True,
                        'caption_font_pt': 10, 'caption_line_baseline_pt': 12,
                        'caption_lines_at_native_width': len(caption_lines),
                        'diagram_plus_caption_and_gap_height_in': native_height + caption_height + 0.15,
                        'text_in_page_and_nodes_checked': True,
                        'system_available_memory_before_render_bytes': available}
        plt.close(fig)
        print(name + ' rendered, min font ' + str(min(font_sizes)), flush=True)

    fig, a, available = page(5.5)
    text(a, 3.35, 5.29, 'FULL local network and training objective', fs=13, weight='bold')
    node(a, .25, 4.35, 2.75, .63, '12 historical rows × 43 channels\nCORE 14 + climate 7 + met 22')
    node(a, 3.70, 4.35, 2.75, .63, 'Climate context: 7 channels\n6 indices + interaction\nLatest eligible period ends t−2')
    node(a, .25, 3.27, 2.75, .74, 'Bidirectional LSTM\n2 layers; 48 units per direction\n12 hidden states hⱼ of 96')
    node(a, 3.70, 3.27, 2.75, .74, 'Climate MLP: 7 → 32 → 16\nEmbedding e with tanh layers')
    arrow(a, (1.625, 4.35), (1.625, 4.01))
    arrow(a, (5.075, 4.35), (5.075, 4.01))
    node(a, .75, 2.15, 5.20, .75, 'Joint attention: [hⱼ, e] → 32 → 1\ntanh, then softmax over 12 historical states\nWeighted hidden mean of 96')
    arrow(a, (1.625, 3.27), (2.4, 2.90))
    arrow(a, (5.075, 3.27), (4.3, 2.90))
    node(a, .75, 1.61, 5.20, .33, 'Concatenate weighted mean 96 + embedding 16 = 112')
    arrow(a, (3.35, 2.15), (3.35, 1.94))
    a.plot([6.45, 6.57, 6.57], [3.64, 3.64, 1.775], color='#444B52', linewidth=1.05)
    arrow(a, (6.57, 1.775), (5.95, 1.775))
    node(a, .25, .96, 2.75, .45, 'Rainfall head\n112 → 64 → 32 → 1')
    node(a, 3.70, .96, 2.75, .45, 'Heavy-event head\n112 → 32 → 1 → sigmoid')
    arrow(a, (2.4, 1.61), (1.625, 1.41))
    arrow(a, (4.3, 1.61), (5.075, 1.41))
    text(a, 3.35, .63, 'Lᵢ = MSE(yᵢᶻ, ŷᵢᶻ) + 0.3 BCE(Hᵢ, p̂ᵢ)\n+ (μᵢ / 2) ‖wᵢ − wʳ‖₂²')
    text(a, 3.35, .205, 'Shared TRAIN input scaling; local target scaling and q80.\nCAAA batch proximity: 0.3μ ≤ μᵢ ≤ μ.')
    save(fig, '01_network', available)

    fig, a, available = page(5.7)
    text(a, 3.35, 5.46, 'CAAA messages in a one-process simulation', fs=13, weight='bold')
    node(a, .25, 4.18, 2.75, 1.00, 'Five simulated clients\nLocal TRAIN labels and updates\nLocal monitor and outer validation\nLocal target scaling and q80')
    node(a, 3.70, 4.18, 2.75, 1.00, 'Parameter server\nAdditive normalized CAAA weights\nSize + monitor performance\n+ fixed TRAIN climate relevance')
    arrow(a, (3.0, 4.80), (3.70, 4.80))
    arrow(a, (3.70, 4.48), (3.0, 4.48))
    text(a, 3.35, 3.78, 'Each round: states, scalar monitoring/validation diagnostics,\nand two fixed whole-TRAIN strengths. Broadcast the global model.')
    node(a, .25, 2.44, 2.75, .93, 'One-time TRAIN summaries\nSample count; input sum\nand squared sum; two local\nENSO/IOD correlations')
    node(a, 3.70, 2.44, 2.75, .93, 'Shared input mean and SD\nClimate correlation summaries\nPlain numerical payloads')
    arrow(a, (3.0, 3.08), (3.70, 3.08))
    arrow(a, (3.70, 2.74), (3.0, 2.74))
    text(a, 3.35, 2.13, 'Upload input statistics and climate summaries.\nBroadcast the input mean and SD once.')
    node(a, .25, 1.31, 6.20, .54, 'Reporting exports: target mean and SD (16 bytes/client).\nTRAIN q80 appears in schema and evaluation diagnostics.')
    text(a, 3.35, .98, 'Primary aggregation coefficients: (1, 0, 0); anchor: (0.3, 0.3, 0.4).')
    text(a, 3.35, .43, 'No secure aggregation or differential privacy is implemented.\nGraph controls separately require aligned TRAIN rainfall sharing.\nCentral and static-router controls use declared raw-data pooling.')
    save(fig, '02_messages', available)

    fig, a, available = page(5.9)
    text(a, 3.35, 5.64, 'Observed input support for target month t', fs=13, weight='bold')
    text(a, 3.35, 5.35, 'Issue: first day of t, at 00:00 UTC')
    timeline = fig.add_axes([.075, .19, .88, .675])
    rows = [('CHIRPS rain for all 12 rows', -24, -2, '#315B7E'),
            ('Six index period ends for all 12 rows', -13, -2, '#71844A'),
            ('Underlying ONI three-month support', -15, -2, '#8B9E65'),
            ('Underlying MEIv2 two-month support', -14, -2, '#A4B185'),
            ('Meteorological months for all 12 rows', -15, -4, '#B28735'),
            ('Historical row origins', -12, -1, '#5C6670')]
    for i, (label, start, end, color) in enumerate(rows):
        y = 5 - i
        timeline.text(-24.6, y + .37, label, fontsize=MIN_FONT, ha='left', va='center', parse_math=False)
        timeline.barh(y, end - start + 1, left=start - .5, height=.35, color=color)
        timeline.text((start + end) / 2, y, f't{start:+d} … t{end:+d}', ha='center', va='center', fontsize=MIN_FONT,
                      color='white' if i in (0, 1, 4, 5) else '#202020', parse_math=False)
    timeline.axvline(0, color='#333333', linestyle='--', linewidth=1.05)
    timeline.set(xlim=(-25, 1.5), ylim=(-.55, 5.65), xlabel='Calendar months relative to forecast target t')
    timeline.set_yticks([])
    timeline.set_xticks([-24, -18, -12, -6, -4, -2, 0], ['t−24', 't−18', 't−12', 't−6', 't−4', 't−2', 't'])
    timeline.spines[['top', 'right', 'left']].set_visible(False)
    timeline.grid(axis='x', alpha=.12)
    timeline.set_axisbelow(True)
    text(a, 3.35, .40, 'Each sequence has 12 rows; sensitivity adds one month of delay.\nSupport is assumed; final archives do not establish operational vintages.\nKnown seasonal harmonics do not use observed target rainfall.')
    save(fig, '03_input_availability', available)

    fig, a, available = page(6.3)
    text(a, 3.35, 6.07, 'Fixed study domains', fs=13, weight='bold')
    map_ax = fig.add_axes([.19, .425, .62, .49])
    domains = [('Client1_NW_Arid', 'C1', 'NW Arid'), ('Client2_Indo_Gangetic', 'C2', 'Indo-Gangetic'),
               ('Client3_NE_Himalaya', 'C3', 'NE Himalaya'), ('Client4_South_Peninsular', 'C4', 'South Peninsular'),
               ('Client5_Central_India', 'C5', 'Central India')]
    require(set(protocol['clients']) == {client for client, _, _ in domains}, 'Frozen five domains differ')

    def polygons(geometry):
        if geometry.is_empty:
            return []
        if geometry.geom_type == 'Polygon':
            return [geometry]
        return [polygon for part in geometry.geoms for polygon in polygons(part)]

    for polygon in polygons(country):
        map_ax.add_patch(Polygon(list(polygon.exterior.coords), facecolor='#F2F2F2', edgecolor='#777777', linewidth=.7))
    for i, ((client, short, name), color) in enumerate(zip(domains, COLORS)):
        south, north, west, east = protocol['clients'][client]
        for polygon in polygons(country.intersection(box(west, south, east, north))):
            map_ax.add_patch(Polygon(list(polygon.exterior.coords), facecolor=color, alpha=.22, edgecolor=color, linewidth=1.1))
        map_ax.add_patch(Rectangle((west, south), east - west, north - south, fill=False, edgecolor=color, linewidth=1.05))
        map_ax.text((west + east) / 2, (south + north) / 2, short, ha='center', va='center',
                    fontsize=11.5, fontweight='bold', color=color, parse_math=False)
        column, row = i % 2, i // 2
        x, y = .35 + column * 3.16, 2.08 - row * .57
        text(a, x, y, f'{short}  {name}', ha='left', weight='bold', color=color, fs=11)
        text(a, x, y - .22, f'{south:g}–{north:g}°N; {west:g}–{east:g}°E', ha='left')
    map_ax.set(xlim=(66.5, 99), ylim=(5.5, 37.5), xlabel='Longitude °E', ylabel='Latitude °N')
    map_ax.set_aspect('equal')
    map_ax.grid(alpha=.12)
    map_ax.spines[['top', 'right']].set_visible(False)
    map_ax.set_xticks([70, 75, 80, 85, 90, 95])
    map_ax.set_yticks([10, 15, 20, 25, 30, 35])
    text(a, 3.51, .94, 'C1 and C5 overlap. The five domains\ndo not partition or cover all India.', ha='left')
    text(a, 3.35, .32, 'Native centers within country and half-open boxes select each mask.\nWhole-cell WGS84 areas supply weights; no grid resampling.')
    save(fig, '04_fixed_domains', available)

    (out / 'captions.json').write_bytes((original / 'captions.json').read_bytes())
    (out / 'captions.txt').write_bytes((original / 'captions.txt').read_bytes())
    (out / 'captions_print_10pt.txt').write_text('\n\n'.join(name + '\n' + '\n'.join(wrap_caption(caption))
                                                         for name, caption in captions.items()), encoding='utf-8')
    (out / 'label_text.json').write_text(json.dumps(all_text, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for name, expected in original_hashes.items():
        require(sha(original / name) == expected and sha(repo / 'docs/method_figures' / name) == expected,
                'Frozen originals or repository copy changed during render')
    manifest = {'status': 'RENDERED_AWAITING_ORIGINAL_RESOLUTION_VISUAL_QA', 'source_only': True,
                'native_width_in': WIDTH, 'minimum_all_text_span_pt': MIN_FONT,
                'evaluation_labels_or_scores_opened': False, 'model_training_or_inference_performed': False,
                'original_v3_preserved': True, 'original_manifest_sha256': sha(original / 'manifest.json'),
                'source_sha256': sources, 'source_paths_read_or_hashed': {key: str(value) for key, value in source_paths.items()},
                'generator_sha256': sha(__file__), 'font_assessment_sha256': sha(out.parent / 'font_assessment.json'),
                'layout': layout, 'files': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                                          for path in sorted(out.iterdir()) if path.is_file()]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'figures': 4, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
