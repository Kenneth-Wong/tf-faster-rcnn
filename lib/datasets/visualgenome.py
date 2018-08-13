import os
import os.path as osp
import pickle
import json
import cv2
import h5py
import numpy as np

from datasets.imdb import imdb
from model.config import cfg
from utils.cython_bbox import bbox_overlaps
from .voc_eval import voc_eval


class visual_genome(imdb):
    def __init__(self, mode, task='det', num_im=-1, num_val_im=5000, filter_empty_rels=True,
                 filter_duplicate_rels=True, filter_non_overlap=True, use_proposals=False):
        imdb.__init__(self, 'visual_genome_%s_%s' % (mode, task))
        if mode not in ('train', 'test', 'val'):
            raise ValueError("Mode must be in test, train or val. Supplied {}".format(mode))
        assert task in ('det', 'rel')
        self.roi_h5 = h5py.File(osp.join(cfg.VG_DIR, 'VG-SGG.h5'), 'r')
        self._info = json.load(open(osp.join(cfg.VG_DIR, 'VG-SGG-dicts.json'), 'r'))
        self._image_set = mode
        self.task = task
        self._image_refs = json.load(open(osp.join(cfg.VG_DIR, 'image_data.json'), 'r'))
        self._roidb_handler = self.gt_roidb

        self.filter_duplicate_rels = filter_duplicate_rels
        self.filter_non_overlap = filter_non_overlap
        self.filter_empty_rels = filter_empty_rels
        self.num_im = num_im
        self.num_val_im = num_val_im
        self.gt_roidb()  # get split_mask and _image_index
        self._filenames = self.load_image_filenames(osp.join(cfg.VG_DIR, 'Images'))
        self._filenames = [self._filenames[i] for i in np.where(self.split_mask)[0]]

        if use_proposals:
            self.rpn_h5 = h5py.File(osp.join(cfg.VG_DIR, 'proposals.h5'), 'r')
            rpn_rois = self.rpn_h5['rpn_rois']
            rpn_scores = self.rpn_h5['rpn_scores']
            rpn_im_to_roi_idx = np.array(self.rpn_h5['im_to_roi_idx'][self.split_mask])
            rpn_num_rois = np.array(self.rpn_h5['num_rois'][self.split_mask])
            self.rpn_rois = []
            for i in range(len(self._filenames)):
                rpn = np.column_stack((rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                                       rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]]))
                self.rpn_rois.append(rpn)
        else:
            self.rpn_rois = None

    def load_image_filenames(self, image_dir):
        corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
        fns = []
        for i, img in enumerate(self._image_refs):
            name = '{}.jpg'.format(img['image_id'])
            if name in corrupted_ims:
                continue
            filename = os.path.join(image_dir, '/'.join(img['url'].split('/')[-2:]))
            if os.path.exists(filename):
                fns.append(filename)
        assert len(fns) == 108073
        return fns

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._filenames[i]

    def load_graphs(self, mode, num_im=-1, num_val_im=0, filter_empty_rels=True, filter_non_overlap=False):
        print('mode==%s' % mode)
        data_split = self.roi_h5['split'][:]
        split = 2 if mode == 'test' else 0
        split_mask = data_split == split

        valid_mask = self.roi_h5['img_to_first_box'][:] >= 0
        valid_mask = np.bitwise_and(split_mask, valid_mask)
        if filter_empty_rels:
            valid_mask &= self.roi_h5['img_to_first_rel'][:] >= 0
        image_index = np.where(valid_mask)[0]

        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0:
            if mode == 'train':
                image_index = image_index[num_val_im:]
            elif mode == 'val':
                image_index = image_index[:num_val_im]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        im_to_first_box = self.roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = self.roi_h5['img_to_last_box'][split_mask]
        all_boxes = self.roi_h5['boxes_%d' % cfg.BOX_SCALE][:]
        labels = self.roi_h5['labels']
        assert (np.all(all_boxes[:, :2] >= 0))
        assert (np.all(all_boxes[:, 2:] > 0))

        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, :2]

        self._info['label_to_idx']['__background__'] = 0
        self._class_to_ind = self._info['label_to_idx']
        self._classes = sorted(self._class_to_ind, key=lambda k: self.class_to_ind[k])
        cfg.ind_to_class = self._classes

        im_to_first_rel = self.roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = self.roi_h5['img_to_last_rel'][split_mask]
        relations = self.roi_h5['relationships'][:]
        relation_predicates = self.roi_h5['predicates'][:, 0]
        assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
        assert (relations.shape[0] == relation_predicates.shape[0])
        self._predicate_to_ind = self._info['predicate_to_idx']
        self._predicate_to_ind['__background__'] = 0
        self._predicates = sorted(self._predicate_to_ind, key=lambda k: self.predicate_to_ind[k])
        cfg.ind_to_predicates = self._predicates

        # get gt roidb and reldb
        gt_roidb = []
        for i in range(len(image_index)):
            assert (im_to_first_box[i] >= 0)
            boxes = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
            gt_classes = labels[im_to_first_box[i]:im_to_last_box[i] + 1, :]

            gt_relations = []
            if im_to_first_rel[i] >= 0:
                predicates = relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
                obj_idx = relations[im_to_first_rel[i]:im_to_last_rel[i] + 1]
                obj_idx = obj_idx - im_to_first_box[i]
                assert (np.all(obj_idx >= 0) and np.all(obj_idx < boxes.shape[0]))
                for j, p in enumerate(predicates):
                    gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])
                gt_relations = np.array(gt_relations)
            else:
                assert not filter_empty_rels
                gt_relations = np.zeros((0, 3), dtype=np.int32)

            if filter_non_overlap:
                inters = bbox_overlaps(boxes, boxes)
                rel_overs = inters[gt_relations[:, 0], gt_relations[:, 1]]
                inc = np.where(rel_overs > 0.0)[0]
                if inc.size > 0:
                    gt_relations = gt_relations[inc]
                else:
                    split_mask[image_index[i]] = 0
                    continue
            overlaps = np.zeros((boxes.shape[0], self.num_classes), dtype=np.float32)
            overlaps[np.arange(boxes.shape[0], dtype=np.int32), gt_classes] = 1.0
            seg_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
            gt_roidb.append({'boxes': boxes, 'gt_classes': gt_classes,
                             'gt_overlaps': overlaps, 'seg_areas': seg_areas,
                             'flipped': False, 'gt_relations': gt_relations})

        image_index = np.where(split_mask)[0] # rebuild image_index
        return gt_roidb, split_mask, image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb, self.split_mask, self._image_index = self.load_graphs(self._image_set, self.num_im,
                                                                        num_val_im=self.num_val_im,
                                                                        filter_empty_rels=self.filter_empty_rels,
                                                                        filter_non_overlap=self.filter_non_overlap)
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)