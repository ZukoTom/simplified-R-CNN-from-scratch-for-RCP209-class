import random
import numpy as np
import tqdm

#image processing stuff

def convert_label_list(*voc_box_coord): # to keep
    """ input format is : voc 2007 format  
    output format is x_mid, y_mid, w, h"""
    ymin, xmin, ymax, xmax = voc_box_coord
    w = xmax - xmin
    h = ymax - ymin
    aire = h * w
    if aire > 1e-3:
        return (xmin + w/2, ymin + h/2, w, h)

def convert_ss_list(ss_roi_coord_list): # not used in the end
    """ input format is : selective search format : xmin, ymin, w, h 
    output format is x_mid, y_mid, w, h"""
    n_list = []
    for coord in ss_roi_coord_list:
        xmin, ymin, w, h = coord
        xmax = xmin + w
        ymax = ymin + h
        aire = (ymax - ymin) * (xmax - xmin)
        if aire > 1e-3:
            n_list.append((xmin + w/2, ymin + h/2, w, h))
    return n_list

#more img processing stuff
def compute_anchors(image):
    """create raw anchors with randomly generated midpoint from a given image, 
    output format is list of (x_mid, y_mid, w, h) tuples
    by design, we create square anchors so its easier to feed into AlexNet"""
    y, x, _ = image.shape
    raw_anchors_list = []
    picks_xmid = random.choices(range(x+1), k=16)
    picks_ymid = random.choices(range(y+1), k=16)
    for idx, (xmid, ymid) in enumerate(zip(picks_xmid, picks_ymid)): # midpoint is a doublet
        if idx % 2 == 0:
            # arbitrary choice that boxes will of dimensions (x+y)/6 x (x+y)/6
            raw_side = int(np.round((x+y)/6)) #length of side of square
            raw_anchors_list.append((xmid, ymid) + (raw_side, raw_side))
        else:# arbitrary choice for bigger boxes 
            raw_side = int(np.round((x+y)/4)) #length of side of square
            raw_anchors_list.append((xmid, ymid) + (raw_side, raw_side))
    return raw_anchors_list

def check_resize_anchors(image, raw_boxes_list):
    """the function takes a list of raw anchor boxes coordinates as input
    together with the corresponding image
    and outputs a list of boxes that don't lie outside the image"""
    y, x, _ = image.shape
    roi_coord_list = []
    for box_coord in raw_boxes_list: #for each anchor
        x_mid, y_mid, w, h = box_coord
        # the if/else sequence is clip+remove outlying coordinates of bboxes to prevent errors etc
        if y_mid - h/2 >= 0 : 
            ymin = y_mid - h/2
        else: #box qui deborde en haut image
            ymin = 1e-3
        if x_mid - w/2 >= 0:
            xmin = x_mid - w/2
        else :
            xmin = 1e-3 #box qui deborde à G
        if y_mid + h/2 <= y:
            ymax = y_mid + h/2
        else : #box qui deborde en bas img
            ymax = y - 1e-3
        if x_mid + w/2 <= x:
            xmax = x_mid + w/2
        else: #box qui deborde à D
            xmax = x - 1e-3
            
        # now reassign w, h because boxes dont lie outside frame anymore
        w = xmax - xmin
        h = ymax - ymin
        x_mid = (xmax+xmin)/2
        y_mid = (ymax+ymin)/2
        
        if 1 - 1e-3 <= h / w <= 1 + 1e-3: #un carré
            crop_img = np.array(image).copy()[
                int(ymin):int(ymax), int(xmin):int(xmax)
                ]
            coord = (x_mid, y_mid, w, h)
            
        elif h / w > 1: #rectangle vertical
            #on trace comme un cercle de rayon w/2 pour reconstruire un carré : 
            #midpoint doit pas bouger selon axe des y
            crop_img = np.array(image).copy()[
                int(y_mid - w/2):int(y_mid + w/2), int(xmin):int(xmax)
                ]
            coord = (x_mid, y_mid, w, w) # xmid est recalculé-modifié
       
        elif h / w < 1: #rectangle horizontal : midpoint doit pas bouger selon les x
            crop_img = np.array(image).copy()[
                int(ymin):int(ymax), int(x_mid - h/2):int(x_mid + h/2)
                ]
            coord = (x_mid, y_mid, h, h) # ymid est recalculé
            
        elif h * w < 1e-3 : #micro boite
            pass

        roi_coord_list.append((crop_img, coord))
    return roi_coord_list 


#some object detection specific functions
def compute_iou(box_p, box_gt):
    """ this function takes two bbox coordinates 4-uplets
    input bbox coordinates format is in xmin, ymin, width, height format
    
    """
    
    xmid_p, ymid_p, w_p, h_p = box_p #coord of proposed box
    xmid_gt, ymid_gt, w_gt, h_gt = box_gt
    aire_p = h_p * w_p 
    aire_gt = h_gt * w_gt

    x_inter_min = max(xmid_p - w_p/2, xmid_gt - w_gt/2)
    x_inter_max = min(xmid_p + w_p/2, xmid_gt + w_gt/2)
    y_inter_min = max(ymid_p - h_p/2, ymid_gt - h_gt/2)
    y_inter_max = min(ymid_p + h_p/2, ymid_gt + h_gt/2)
    
    #if no intersection
    if ((ymid_p - h_p/2) > (ymid_gt + h_gt/2)\
    or (ymid_gt - h_gt/2) > (ymid_p + h_p/2)\
    or (xmid_p - w_p/2) > (xmid_gt + w_gt/2)\
    or (xmid_gt - w_gt/2) > (xmid_p + w_p/2)):
        IoU = 0
    else : # if intersection
        inter = (y_inter_max - y_inter_min)\
                 * (x_inter_max - x_inter_min)
        IoU = float(inter / (aire_p + aire_gt - inter))
        
        coord_inter = (x_inter_min + x_inter_max)/2,\
        (y_inter_min + y_inter_max)/2,\
        x_inter_max - x_inter_min,\
        y_inter_max - y_inter_min
        
    if IoU > 1e-5:
        return IoU, coord_inter
    else:
        return 0, None


def matching_boxes_new(liste_p, liste_gt, iou_match=.3): 
    """pour une img, liste_p est une liste des coord de roi, 
    liste_gt une liste des coord des GT 
    output est une liste des ROI pertinentes, une par ROI le cas echeant"""
    list_relevant = [] # on met dans cette liste les match (pour une img)
    for idx_p, proposed_box in enumerate(liste_p): # pour une proposition de region...
        matches = [] # ...on met dans cette liste : match par proposed box
        for idx_g, gt_box in enumerate(liste_gt): 
            if compute_iou(proposed_box, gt_box)[0] >= iou_match: #si on a un match
                matches.append(
                    ((idx_p, idx_g),
                     compute_iou(proposed_box, gt_box)[0],
                     proposed_box)
                )
        if len(matches) == 1:
            list_relevant.append(matches)
        elif len(matches) > 1: #ici on veut garder que le meilleur match pour une ROI si +sieurs
            multiple_m = []
            for m in matches:
                multiple_m.append(m[1])
            rang = np.argmax(multiple_m)
            list_relevant.append(matches[rang])
            
    return list_relevant
    

def capture_background(liste_p, liste_gt, iou_match=0): #objectness 
    """pour une img, liste_p est une liste des coord de roi, 
    liste_gt une liste des coord des GT associée
    fn retourne, pr une img, liste ROI qui n'ont pas d'intersection avec AUCUNE GT de l'img"""
    list_back = []
    for idx_p, proposed_box in enumerate(liste_p): #pour chaque ROI d'une img
        back_candidate = [] #une liste de back par ROI
        for idx_g, gt_box in enumerate(liste_gt): #on calcule avec toutes les GT possibles
            if compute_iou(proposed_box, gt_box)[0] <= iou_match : #iou match = 0
                back_candidate.append(
                    ((idx_p, idx_g), 
                     compute_iou(proposed_box, gt_box)[0], 
                     proposed_box)
                )
        if len(back_candidate) == len(liste_gt):#si aucune intersection avec GT box
            list_back.append(random.choice(back_candidate)) # un back par ROI maximum

    if len(list_back) > 0:
        if len(list_back) < 2:
            return list_back
        elif len(list_back) >= 2:
            return random.sample(list_back, 2) # 2 back max par img
        


def custom_recall(liste_tuples_pred, liste_tuples_gt, iou_match=.25, iou_info=False): 
    """POUR UNE IMG, liste_tuples_pred est une liste des propositions, coord de proposition; 
    liste_tuples_gt une liste des GT_label, coord des GT; 
    output est un score de recall pour 1 img, listes des IoU si TP """
    list_score_iou = []
    TP = 0 
    num_gt = len(liste_tuples_gt)
    for gt_label, gt_coord in liste_tuples_gt: # on itere sur les GT car perspective de mesure d'erreur
        matches = [] 
        for pred_label, pred_box in liste_tuples_pred: 
            #si on a un match
            if compute_iou(gt_coord, pred_box)[0] >= iou_match and int(gt_label) == int(pred_label): 
                matches.append(compute_iou(pred_box, gt_coord)[0])
                
        if len(matches) > 0: #pour une GT peut y avoir qu'une seule bonne pred
            rang = np.argmax(matches)
            list_score_iou.append(matches[rang])

            TP += 1
        recall = TP / num_gt
    if iou_info == True:
        return recall, list_score_iou
    else:
        return recall


def bounding_box_coord_loss(sample_image, pred_coord_list, gt_coord_list):
    for pc in pred_coord_list:
        xmid_p, ymid_p, w_p, h_p = pc
        xmid_p, ymid_p, w_p, h_p = pc    