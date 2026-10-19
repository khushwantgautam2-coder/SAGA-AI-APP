"""Batch inference and an in-memory Excel report for both Streamlit apps."""

from io import BytesIO
import hashlib
import logging

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import xlsxwriter

from cvms_core import compute_cvms_score, run_inference

LANDMARK_NAMES = (
    'C2a', 'C2p', 'C2d', 'C3ua', 'C3up', 'C3la', 'C3lp',
    'C4ua', 'C4up', 'C4la', 'C4lp', 'C4um', 'C4am',
)
RESULT_COLUMNS = ['Image ID', 'Filename', 'Status', 'CVMS score', 'Stage', 'Error']
POINT_COLUMNS = ['Image ID', 'Filename', 'Index', 'Landmark', 'Crop X', 'Crop Y', 'Image X', 'Image Y']


def process_batch(uploads, net1, net2, progress=None):
    """Keep one result per upload, including duplicate names and failures."""
    rows, points = [], []
    for image_id, upload in enumerate(uploads, 1):
        row = dict(zip(RESULT_COLUMNS, [image_id, upload.name, 'Failed', None, None, '']))
        try:
            data = upload.getvalue()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
            if image is None:
                raise ValueError('Image could not be decoded. Use a valid JPG or PNG file.')
            result = run_inference(net1, net2, image)
            if result is None:
                raise ValueError('No valid C2–C4 crop was detected.')
            predicted = np.asarray(result['pred_pts'])
            if predicted.shape != (13, 2) or not np.isfinite(predicted).all():
                raise ValueError('The model returned invalid landmark coordinates.')
            score, stage = compute_cvms_score(result['pred_pts'])
            if not np.isfinite(score):
                raise ValueError('The model returned an invalid CVMS score.')
            x1, y1, _, _ = result['bbox']
            image_points = []
            for index, ((x, y), name) in enumerate(zip(result['pred_pts'], LANDMARK_NAMES)):
                image_points.append([image_id, upload.name, index, name, int(x), int(y), int(x + x1), int(y + y1)])
            points.extend(image_points)
            row.update({'Status': 'Success', 'CVMS score': float(score), 'Stage': f'CS{stage}'})
        except ValueError as exc:
            row['Error'] = str(exc)
        except Exception:
            logging.exception('Batch inference failed for image ID %s', image_id)
            row['Error'] = 'Inference failed for this image. Check the application logs.'
        rows.append(row)
        if progress:
            progress(image_id / len(uploads), f'Processed {image_id} of {len(uploads)} images')
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), pd.DataFrame(points, columns=POINT_COLUMNS)


def excel_report(results, landmarks):
    """Return XLSX bytes; filenames remain literal text, never Excel formulas."""
    report = results[['Filename', 'Stage', 'CVMS score']].copy()
    report['Stage'] = report['Stage'].map(lambda value: int(str(value).removeprefix('CS')) if pd.notna(value) else None)
    report = report.rename(columns={'CVMS score': 'QCVMS'})
    output = BytesIO()
    with xlsxwriter.Workbook(output, {'in_memory': True, 'strings_to_formulas': False, 'strings_to_urls': False}) as workbook:
        header = workbook.add_format({'font_name': 'Calibri', 'font_size': 11})
        number = workbook.add_format({'num_format': '0.0000'})
        failures = results.loc[results['Status'] != 'Success', ['Image ID', 'Filename', 'Error']]
        sheets = [('Results', report), ('Landmarks', landmarks)]
        if not failures.empty:
            sheets.append(('Errors', failures))
        for name, frame in sheets:
            sheet = workbook.add_worksheet(name)
            sheet.freeze_panes(1, 2)
            sheet.write_row(0, 0, list(frame.columns), header)
            for r, values in enumerate(frame.itertuples(index=False, name=None), 1):
                for c, value in enumerate(values):
                    if pd.isna(value):
                        continue
                    if isinstance(value, str):
                        sheet.write_string(r, c, value)
                    else:
                        sheet.write(r, c, value, number if frame.columns[c] == 'QCVMS' else None)
            if name != 'Results':
                sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            for c, column in enumerate(frame.columns):
                sheet.set_column(c, c, 20 if column == 'Filename' else 65 if column == 'Error' else 12)
        summary = workbook.add_worksheet('Summary')
        successful = int((results['Status'] == 'Success').sum())
        values = [
            ('SAGA-AI batch report', 'Value'),
            ('Images uploaded', len(results)), ('Successful predictions', successful),
            ('Failed images', len(results) - successful),
            *[(f'CS{stage} predictions', int((results['Stage'] == f'CS{stage}').sum())) for stage in range(1, 5)],
            ('Coordinates', 'Crop coordinates are relative to the detected crop; Image coordinates are relative to the uploaded image.'),
            ('Landmark index', 'Zero-based; the same order as the original model.'),
            ('Stage thresholds', 'CS1 < 1.7404; CS2 < 2.623; CS3 < 3.5199; otherwise CS4.'),
            ('Score calculation', 'Original Chen formula implementation; both ratios clamped to [0.1, 4.0].'),
            ('Failed predictions', 'Filename is retained in Results with blank Stage and QCVMS; see Errors for the reason.'),
        ]
        for r, values_row in enumerate(values):
            summary.write_row(r, 0, values_row, header if r == 0 else None)
        summary.set_column(0, 0, 30)
        summary.set_column(1, 1, 72, workbook.add_format({'text_wrap': True, 'valign': 'top'}))
        for r in range(8, 13):
            summary.set_row(r, 32)
        summary.freeze_panes(1, 0)
    return output.getvalue()


def render_batch_processing(net1_path, net2_path, models_ready, load_net1, load_net2):
    st.subheader('Batch Processing')
    st.caption('Upload full lateral cephalometric X-rays. Ground-truth annotations are not required.')
    uploads = st.file_uploader('Upload images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key='batch_images')
    # Persist a finished run across download/rerender, but never display results for a different selection.
    signature = hashlib.sha256(b'filename-stage-qcvms-v2')
    for item in uploads:
        signature.update(item.name.encode())
        signature.update(hashlib.sha256(item.getvalue()).digest())
    signature.update(str((net1_path, net2_path)).encode())
    run_id = signature.hexdigest()
    if st.session_state.get('batch_run_id') != run_id:
        st.session_state.pop('batch_report', None)
        st.session_state['batch_run_id'] = run_id
    if not models_ready:
        st.warning('Model weights are required before processing images.')
    if uploads:
        st.caption(f'{len(uploads)} images selected. Images are processed one at a time.')
    if st.button('Process batch', disabled=not (uploads and models_ready), key='process_batch'):
        st.session_state.pop('batch_report', None)
        try:
            net1, net2 = load_net1(net1_path), load_net2(net2_path)
        except Exception:
            logging.exception('Could not load batch models')
            st.error('Could not load the model weights. Check the configuration and application logs.')
            return
        progress = st.progress(0.0, text='Starting batch...')
        results, landmarks = process_batch(uploads, net1, net2, lambda fraction, text: progress.progress(fraction, text=text))
        progress.empty()
        st.session_state['batch_report'] = (results, excel_report(results, landmarks))
    report = st.session_state.get('batch_report')
    if report is not None:
        results, workbook = report
        successful = int((results['Status'] == 'Success').sum())
        a, b, c = st.columns(3)
        a.metric('Images', len(results))
        b.metric('Successful', successful)
        c.metric('Failed', len(results) - successful)
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.download_button('Download results as Excel', data=workbook, file_name='saga_ai_batch_results.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='batch_excel')
        if st.button('Clear batch results'):
            st.session_state.pop('batch_report', None)
            st.rerun()
