import numpy as np

def precision_score(test_y, pred_y, k=1):
    p_score = []
  
    for i in xrange(len(test_y)):
        # print(pred_y[i][-k:])
        # print(pred_y[i][-k])
        # result_at_topk = pred_y[i][-k:]
        count = 0
        for j in range(0,k):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        p_score.append(float(count) / float(k))
            # if j in test_y[i]:
                # count += 1
        # p_score.append(float(count) / float(k))

    return np.mean(p_score)

def recall_score(test_y, pred_y, k=1):
    r_score = []
    for i in xrange(len(test_y)):
        # result_at_topk = pred_y[i][-k:]
        # count = 0
        # for j in result_at_topk:
        #     if j in test_y[i]:
        #         count += 1
        count = 0
        for j in range(0,k):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        r_score.append(float(count) / float(len(test_y[i])))

    return np.mean(r_score)

def hits_score(test_y, pred_y, k=1):
    h_score = []
    for i in xrange(len(test_y)):
        # result_at_topk = pred_y[i][-k:]
        # count = 0
        # for j in result_at_topk:
        #     if j in test_y[i]:
        #         count += 1
        count = 0
        for j in range(0,k):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        h_score.append(1 if count > 0 else 0)

    return np.mean(h_score)