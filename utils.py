import random
import numpy as np

#image processing stuff

def convert_label_list(*t_uplet): # to keep
    """ input format is : voc 2007 format  
    output format is x_mid, y_mid, w, h"""
    ymin, xmin, ymax, xmax = t_uplet
    w = xmax - xmin
    h = ymax - ymin
    aire = (ymax - ymin) * (xmax - xmin)
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
    for i, e in enumerate(zip(picks, picks2)): # i is a doulblet
        if i % 2 == 0:
            # arbitrary choice that boxes will of dimensions (x+y)/5 x (x+y)/5
            une_liste.append(e + (int(np.round((x+y)/6)), int(np.round((x+y)/6))))
        else:
            une_liste.append(e + (int(np.round((x+y)/4)), int(np.round((x+y)/4))))
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
        #cropping : note from here we need to ditch using xmid, ymid, w, h since we dont know if box been clipped or not
        if 1 - 1e-3 <= (ymax - ymin) / (xmax - xmin) <= 1 + 1e-3:
            crop_img = np.array(imag).copy()[int(ymin):int(ymax), int(xmin):int(xmax)]
            coord = ((xmax+xmin)/2, (ymax+ymin)/2, xmax - xmin, ymax-ymin)
            
        elif (ymax - ymin) / (xmax - xmin) > 1: #rectangle vertical
            #on trace comme un cercle de rayon w/2 pour reconstruire un carré : midpoint doit pas bouger selon axe des y
            crop_img = np.array(imag).copy()[int((ymax+ymin)/2 - (xmax-xmin)/2):int((ymax+ymin)/2 + (xmax-xmin)/2), int(xmin):int(xmax)]
            
            coord = ((xmax+xmin)/2, (ymax+ymin)/2, xmax - xmin, xmax - xmin) # xmid est recalculé-modifié
       
        elif(ymax - ymin) / (xmax - xmin) < 1: #rectangle horizontal : midpoint doit pas bouger selon les x
            
            crop_img = np.array(imag).copy()[int(ymin):int(ymax), int((xmax+xmin)/2 - (ymax-ymin)/2):int((xmax+xmin)/2 + (ymax-ymin)/2)]
            coord = ((xmax+xmin)/2, (ymax+ymin)/2, ymax-ymin, ymax-ymin) # ymid est recalculé
            
        elif (ymax - ymin) * (xmax - xmin) < 1e-3:
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
    
    x_p, y_p, w_p, h_p = box_p #coord of predicted box
    x_gt, y_gt, w_gt, h_gt = box_gt
    aire_p = h_p * w_p # midpoint not needed BTW
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



def matching_boxes(liste_p, liste_gt, iou_match=.2): #objectness 
    """pour une img, liste_p est une liste des coord de roi, liste_gt une liste des coord des GT associée
    cette fn s'applique pour ces deux listes, output est une liste des intersections au dela d'un objectness threshold fixé,
    une autre si l'inter est negligeable ou nulle"""
    matches = []
    back = []
    indices_p = []
    indices_back = []
    for idx, i in enumerate(liste_p): #expect often more pred than gt boxes
        for idx2, j in enumerate(liste_gt): #idx2 donne le rang du label
            if compute_iou(i, j)[0] >= iou_match : #on a rarement de bon IoU donc on met le seuil bas
                matches.append(((idx, idx2), np.round(compute_iou(i, j)[0], 2), i)) #on recupere i cad : coord de la pred
                indices_p.append(idx)
            elif compute_iou(i, j)[0] == 0 :#et les background combinaisons de la même pred et une autre gt bbox
                back.append(((idx, idx2), np.round(compute_iou(i, j)[0], 2), i))
                indices_back.append(idx)
            else:
                pass
                
    if len(back) >= 2:
        return matches, random.sample(back, 2)
    else:
        return matches, back






