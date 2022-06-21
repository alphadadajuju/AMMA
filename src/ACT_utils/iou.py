import torch
import numpy as np

def intersection_over_union_tubelet_association(boxes1, boxes2, classes1, classes2, K, iou_thresh=0.75):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    
    batch, N, boxes = boxes1.shape
    
    assert boxes // 4 == K, "K is not set properly."
        
    iou_table = np.ones((N, N)) * -1
    
    
    for i in range(0,N):
        classes1_q = classes1[:,i,:] # b,1
        classes1_q = classes1_q.squeeze(dim=0).unsqueeze(1)
        
        classes1_q = classes1_q.repeat(N,1)
        
        is_qk_same = (classes1_q == classes2.squeeze(0)).type(torch.cuda.FloatTensor)
        
        boxes1_q = boxes1[0,i:i+1,:]
        boxes1_q = boxes1_q.repeat(N,1)
        boxes2_k = boxes2.squeeze(0)
        
        iou = 0.0
        for k in range(0,K):
            # variable names are kept the same as the original code
            box1_x1 = boxes1_q[..., 0+k*4:1+k*4]
            box1_y1 = boxes1_q[..., 1+k*4:2+k*4]
            box1_x2 = boxes1_q[..., 2+k*4:3+k*4]
            box1_y2 = boxes1_q[..., 3+k*4:4+k*4]
            box2_x1 = boxes2_k[..., 0+k*4:1+k*4]
            box2_y1 = boxes2_k[..., 1+k*4:2+k*4]
            box2_x2 = boxes2_k[..., 2+k*4:3+k*4]
            box2_y2 = boxes2_k[..., 3+k*4:4+k*4]

            x1 = torch.max(box1_x1, box2_x1)
            y1 = torch.max(box1_y1, box2_y1)
            x2 = torch.min(box1_x2, box2_x2)
            y2 = torch.min(box1_y2, box2_y2)

            # Need clamp(0) in case they do not intersect, then we want intersection to be 0
            intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
            box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
            box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
            
            iou += (intersection / (box1_area + box2_area - intersection + 1e-6) * is_qk_same)
        avg_iou = iou / float(K)
        iou_table[i,:] = avg_iou.squeeze(1).cpu().numpy()
    
    #positive_mask = (iou_table > iou_thresh).astype(float)
    positive_mask1 = (iou_table < iou_thresh).astype(float)
    positive_mask2 = (iou_table > 0.4).astype(float)
    iou_table_masked = iou_table * positive_mask1 * positive_mask2
    
    # policy: for each box1, find its best match in box2
    max_iou = np.max(iou_table, axis=1) # keep a record which ones NOT to be augmented 
    max_iou_arg = np.argmax(iou_table_masked, axis=1)
    
    boxes1_has_match = (max_iou > 0.4)
    
    boxes1c = boxes1.clone().squeeze(0).cpu().numpy() 
    boxes2c = boxes2.clone().squeeze(0).cpu().numpy() 
    boxes12 = np.zeros((N,4*K))
    
    coe1 = np.array([0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1, 1, 1 ,1])
    coe2 = 1 - coe1
    for i in range(0, N): # consistent detection; no need to refine
        if max_iou[i] > iou_thresh:
            boxes12[i,:] = boxes1c[i,:] 
            
        else: # not so confident; need to be refined
            #boxes12[i,:] = (boxes1c[i,:] + boxes2c[max_iou_arg[i], :]) / 2.0
            boxes12[i,:] = (boxes1c[i,:]*coe1 + boxes2c[max_iou_arg[i], :]*coe2)
            #print(i)
    
    boxes12_tensor = torch.from_numpy(boxes12).unsqueeze(0).type(torch.cuda.FloatTensor)
    valid_ind = np.argwhere(boxes1_has_match == True)
    return boxes12_tensor, valid_ind

    #iou_sum = np.sum(iou_table_masked, axis=1, keepdims=True)
    #boxes1_has_match = (iou_sum > 0.0)
    
    #valid_ind = np.argwhere(boxes1_has_match == True)
    #return valid_ind
    

    '''
    # TODO: this can probably be optimized by parallizing 
    for i in range(0,N):
        for j in range(0,N):
            
            classes1_q = classes1[:,i,:] # b,1
            classes2_k = classes2[:,j,:]
            
            is_qk_same = (classes1_q == classes2_k).type(torch.cuda.FloatTensor)
            
            boxes1_q = boxes1[:,i,:] # b,4K
            boxes2_k = boxes2[:,j,:]
            
            iou = 0.0
            for k in range(0,K):
                # variable names are kept the same as the original code
                box1_x1 = boxes1_q[..., 0+k*4:1+k*4]
                box1_y1 = boxes1_q[..., 1+k*4:2+k*4]
                box1_x2 = boxes1_q[..., 2+k*4:3+k*4]
                box1_y2 = boxes1_q[..., 3+k*4:4+k*4]
                box2_x1 = boxes2_k[..., 0+k*4:1+k*4]
                box2_y1 = boxes2_k[..., 1+k*4:2+k*4]
                box2_x2 = boxes2_k[..., 2+k*4:3+k*4]
                box2_y2 = boxes2_k[..., 3+k*4:4+k*4]
    
                x1 = torch.max(box1_x1, box2_x1)
                y1 = torch.max(box1_y1, box2_y1)
                x2 = torch.min(box1_x2, box2_x2)
                y2 = torch.min(box1_y2, box2_y2)
    
                # Need clamp(0) in case they do not intersect, then we want intersection to be 0
                intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
                box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
                box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
                
                iou += (intersection / (box1_area + box2_area - intersection + 1e-6) * is_qk_same)
            
            avg_iou = iou / float(K)
            
            iou_table[:,i,j] = avg_iou.squeeze().cpu().numpy()
            
    positive_mask = (iou_table > iou_thresh).astype(float)
    iou_table_masked = iou_table * positive_mask
    
    iou_sum = np.sum(iou_table_masked, axis=2, keepdims=True)
    boxes1_has_match = (iou_sum > 0.0)
    
    valid_ind = np.argwhere(boxes1_has_match == True)
    return valid_ind
    '''
def intersection_over_union_orig(boxes_preds, boxes_labels, box_format="corners"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)