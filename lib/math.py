def calculate_distance(x0, y0, x1, y1):
    return abs(((x0 - x1) * (x0 - x1)) + ((y0 - y1) * (y0 - y1)))

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
     
    return intersection / union  

def compute_jaccard_similarity(data):
    similarity_dict = {}
    grouped = data.groupby('order_id')['item_id'].apply(set)
    for order_dum, items in grouped.items():
        similarities = []
        for other_order_dum, other_items in grouped.items():
            if order_dum == other_order_dum:
                similarities.append(1.0)  # similarity with itself is 1
            else:
                similarity = jaccard_similarity(items, other_items)
                similarities.append(similarity)
        similarity_dict[order_dum] = similarities
    return grouped, similarity_dict