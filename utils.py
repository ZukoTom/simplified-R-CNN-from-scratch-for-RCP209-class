import random
import numpy as np
import tqdm

#image processing stuff

def convert_label_list(*t_uplet): # to keep
    """ input format is : voc 2007 format  
    output format is x_mid, y_mid, w, h"""
    ymin, xmin, ymax, xmax = t_uplet
    w = xmax - xmin
    h = ymax - ymin
    aire = h * w
    if aire > 1e-3:
        return (xmin + w/2, ymin + h/2, w, h)

def convert_ss_list(lisst): # not used in the end
    """ input format is : selective search format : xmin, ymin, w, h 
    output format is x_mid, y_mid, w, h"""
    n_list = []
    for i in lisst:
        xmin, ymin, w, h = i
        xmax = xmin + w
        ymax = ymin + h
        aire = (ymax - ymin) * (xmax - xmin)
        if aire > 1e-3:
            n_list.append((xmin + w/2, ymin + h/2, w, h))
    return n_list


#more img processing stuff
def compute_anchors(imag):
    """create anchors with randomly generated midpoint, output format is x_mid, y_mid, w, h
    by design, we create square anchors so its easier to feed into AlexNet later"""
    y, x, _ = imag.shape
    une_liste = []
    picks = random.choices(range(x+1), k=16)
    picks2 = random.choices(range(y+1), k=16)
    for idx, midpoint in enumerate(zip(picks, picks2)): # i is a doulblet
        if idx % 2 == 0:
            # arbitrary choice that boxes will of dimensions (x+y)/6 x (x+y)/6
            une_liste.append(midpoint + (int(np.round((x+y)/6)), int(np.round((x+y)/6))))
        else:
            une_liste.append(midpoint + (int(np.round((x+y)/4)), int(np.round((x+y)/4))))
    return une_liste


#more img processing stuff
def produce_crop_list(imag, lisht):
    """2nd argument "lisht" is a list of box coordinates in following format : (xmid, ymid, weight, height)
    returns a list of images
    Don't be puzzled! since we clip fraction of boxes outside img, we use xmax, ymax when calculating output"""
    y, x, _ = imag.shape
    crop_coord = []
    liste_crop = []
    ultimate_list = []
    for i in lisht: #for anchor
        x_mid, y_mid, w, h = i
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
        if 1 - 1e-3 <= h / w <= 1 + 1e-3: #un carré
            crop_img = np.array(imag).copy()[int(ymin):int(ymax), int(xmin):int(xmax)]
            coord = ((xmax+xmin)/2, (ymax+ymin)/2, w, h)
            
        elif h / w > 1: #rectangle vertical
            #on trace comme un cercle de rayon w/2 pour reconstruire un carré : midpoint doit pas bouger selon axe des y
            crop_img = np.array(imag).copy()[int((ymax+ymin)/2 - (xmax-xmin)/2):int((ymax+ymin)/2 + (xmax-xmin)/2), int(xmin):int(xmax)]
            
            coord = ((xmax+xmin)/2, (ymax+ymin)/2, w, w) # xmid est recalculé-modifié
       
        elif h / w < 1: #rectangle horizontal : midpoint doit pas bouger selon les x
            
            crop_img = np.array(imag).copy()[int(ymin):int(ymax), int((xmax+xmin)/2 - (ymax-ymin)/2):int((xmax+xmin)/2 + (ymax-ymin)/2)]
            coord = ((xmax+xmin)/2, (ymax+ymin)/2, h, h) # ymid est recalculé
            
        elif h * w < 1e-3 : #micro boite
            pass
        else:
            pass

        ultimate_list.append((crop_img, coord))
    return ultimate_list 


#some object detection specific functions
def compute_iou(box_p, box_gt):
    """ this function takes two bbox coordinates 4-uplets
    input bbox coordinates format is in xmin, ymin, width, height format
    
    """
    
    x_p, y_p, w_p, h_p = box_p #coord of proposed box
    x_gt, y_gt, w_gt, h_gt = box_gt
    aire_p = h_p * w_p 
    aire_gt = h_gt * w_gt

    x_inter_min = max(x_p - w_p/2, x_gt - w_gt/2)
    x_inter_max = min(x_p + w_p/2, x_gt + w_gt/2)
    y_inter_min = max(y_p - h_p/2, y_gt - h_gt/2)
    y_inter_max = min(y_p + h_p/2, y_gt + h_gt/2)
    
    #if no intersection
    if ((y_p - h_p/2) > (y_gt + h_gt/2)\
    or (y_gt - h_gt/2) > (y_p + h_p/2)\
    or (x_p - w_p/2) > (x_gt + w_gt/2)\
    or (x_gt - w_gt/2) > (x_p + w_p/2)):
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
    """pour une img, liste_p est une liste des coord de roi, liste_gt une liste des coord des GT 
    output est une liste des intersections au dela d'un objectness threshold fixé"""
    some_list = [] # on met dans cette liste les match (pour une img)
    for idx_p, proposed_box in enumerate(liste_p): # pour une proposition de region...
        matches = [] # ...on met dans cette liste les match
        for idx_g, gt_box in enumerate(liste_gt): 
            if compute_iou(proposed_box, gt_box)[0] >= iou_match: #si on a un match
                matches.append(((idx_p, idx_g), compute_iou(proposed_box, gt_box)[0], proposed_box))
        if len(matches) > 0: #ici on veut garder que le meilleur match pour une ROI si +sieurs
            multiple_m = []
            for m in matches:
                multiple_m.append(m[1])
            rang = np.argmax(multiple_m)
            some_list.append(matches[rang])
            
    return some_list
    

def capture_background(liste_p, liste_gt, iou_match=0.1): #objectness 
    """pour une img, liste_p est une liste des coord de roi, liste_gt une liste des coord des GT associée
    cette fn liste, pr une img, les ROI qui n'ont pas d'intersection avec AUCUNE GT de l'img"""
    list_back = []
    for idx_p, proposed_box in enumerate(liste_p): #pour chaque ROI d'une img
        back_candidate = [] #une liste par ROI
        for idx_g, gt_box in enumerate(liste_gt): #on calcule avec toutes les GT possibles
            if compute_iou(proposed_box, gt_box)[0] <= iou_match : #iou match = 0
                back_candidate.append(((idx_p, idx_g), compute_iou(proposed_box, gt_box)[0], proposed_box))
        if len(back_candidate) == len(liste_gt):
            list_back.append(random.choice(back_candidate))
        
    if len(list_back) >= 2:
        return random.sample(list_back, 2)
    else:
        return list_back


def custom_recall(liste_tuples_pred, liste_tuples_gt, iou_match=.25): 
    """POUR UNE IMG, liste_tuples_pred est une liste des GT, coord de GT_label; 
    liste_tuples_gt une liste des ROI_label, coord des ROI; 
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
                
        if len(matches) > 0: #pour une ROI peut y avoir qu'une seule bonne pred
            multiple_m = []
            for m in matches:
                multiple_m.append(m)
            rang = np.argmax(multiple_m)
            list_score_iou.append(matches[rang])

            TP += 1
        recall = TP / num_gt
    return recall, list_score_iou
