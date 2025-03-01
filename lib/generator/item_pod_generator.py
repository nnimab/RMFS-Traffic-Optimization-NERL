import os
import numpy as np
import pandas as pd

from lib.file import *
from lib.constant import *

def config_items_slots(dev_mode=False):
    working_path = get_working_path(dev_mode)

    print("Setting up item slot configuration based on Item and Pod dictionary (see data/v2/ folder)...")

    # check if file not exists
    items_slots_path = os.path.join(PARENT_DIRECTORY, 'data/output/items_slots_configuration.csv')
    if not os.path.exists(items_slots_path):

        pods_dictionary_path = os.path.join(PARENT_DIRECTORY, 'data/input/pods_dictionary.csv')
        pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

        item_path = os.path.join(PARENT_DIRECTORY, 'data/input/items_dictionary.csv')
        item_df = pd.read_csv(item_path, index_col=False)
        item_df = item_df[["item_code", "box_volume",
                           "item_volume", "number_of_item_in_a_box"]].copy()

        # getl slot type from pod_dictionary
        items_slots_configuration = pd.DataFrame()
        slot_types = np.sort(pods_dictionary["slot_type"].unique())
        for slot_type in slot_types:
            # get slot volume
            slot_volume = pods_dictionary.loc[pods_dictionary["slot_type"]
                                     == slot_type, "slot_volume"].values[0]
            item_df["slot_type"] = slot_type
            item_df["slot_volume"] = slot_volume
            item_df["max_box_in_slot"] = (
                slot_volume / item_df["box_volume"]).astype(int)
            item_df["max_item_in_slot"] = item_df["max_box_in_slot"] * \
                item_df["number_of_item_in_a_box"]

            items_slots_configuration = pd.concat(
                [items_slots_configuration, item_df], axis=0)

        items_slots_configuration.to_csv(items_slots_path, index=False)

    else:
        print("    Item slot configuration already exists. Delete the file to reconfigure.")
        print()

def check_items_pods_feasibility(total_sku, pod_types, pods_dictionary):
    feasible = True
    for pod_type in pod_types:
        total_slot = pods_dictionary.loc[pods_dictionary["pod_type"] == pod_type].shape[0]
        if  total_slot > total_sku:
            print("Pod type", pod_type, "has more slot ("+str(total_slot)+") than total SKU ("+str(total_sku)+").")
            print("In this system we asume that each pod cannot store same item in multiple slot.")
            print("Please adjust the number of items before we can continue to generate the items and pods.")
            feasible = False
            break
    return feasible

def gen_items(pod_types=[0], 
              total_sku=500, 
              items_class_conf={"A": 0.1, "B": 0.3, "C": 0.6}, 
              items_pods_inventory_levels={"A": 0.3, "B": 0.4, "C": 0.5}, 
              items_warehouse_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6},
              select_option=1, 
              dev_mode=False):

    # Generate or select items based on class of items.
    # The items must align with the designated slot type within the pod.

    # get the working path
    working_path = get_working_path(dev_mode)

    # get item slot configuration
    items_slots_configuration_path = os.path.join(PARENT_DIRECTORY, 'data/output/items_slots_configuration.csv')
    items_slots_configuration = pd.read_csv(items_slots_configuration_path, index_col=False)

    # get pod configuration
    pods_dictionary_path = os.path.join(PARENT_DIRECTORY, 'data/input/pods_dictionary.csv')
    pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

    if check_items_pods_feasibility(total_sku, pod_types, pods_dictionary):
        
        # get list of item id that can be stored in the pod type (and slot type)
        item_code_arr = list()
        for pod_type in pod_types:
            slot_types = pods_dictionary.loc[pods_dictionary["pod_type"]
                                    == pod_type, "slot_type"].unique()
            slot_types = np.sort(slot_types)

            # print(slot_types)
            for slot_type in slot_types:
                # print(slot_type)
                item_code = items_slots_configuration.loc[(items_slots_configuration["slot_type"] == slot_type) & (
                    items_slots_configuration["max_box_in_slot"] > 0), "item_code"]
                item_code_arr = item_code_arr + item_code.to_list()

        # remove duplicates
        item_code_arr = np.unique(item_code_arr)

        # retrieve item details corresponding to the list of item IDs compatible with the pod's slot specifications
        item_path = os.path.join(PARENT_DIRECTORY, 'data/input/items_dictionary.csv')
        item_df = pd.read_csv(item_path, index_col=False)

        item_df = item_df.loc[item_df["item_code"].isin(item_code_arr)].copy()
        item_df = item_df[["item_code", "item_class", "item_order_frequency", "item_initial_quantity_inventory", "box_length", "box_width", "box_height", "box_volume", "box_weight", "number_of_item_in_a_box",
                           "item_volume", "item_unit", "item_quantity_order_unique"]].copy()

        # select items according to the class and its proportion
        items = pd.DataFrame()
        for class_name, conf in items_class_conf.items():

            # get list of the item's code based on the class
            item = item_df.loc[item_df["item_class"] == class_name].copy()
            item = item.sort_values(by="item_order_frequency", ascending=False)

            # print(items_inventory_levels)
            item["item_pod_inventory_level"] = items_pods_inventory_levels[class_name]
            item["item_warehouse_inventory_level"] = items_warehouse_inventory_levels[class_name]

            # calculate the probability of order frequency for each item
            item_probability = item["item_order_frequency"] / \
                item["item_order_frequency"].sum()

            # select option 1: select items based on the probability of order frequency for each class.
            # else: select items based on the top n-percent of the class ratio
            if select_option == 1:

                # select items based on the probability of order frequency for each class.
                item_code = np.random.choice(item["item_code"].to_list(), size=int(
                    total_sku*conf), p=item_probability, replace=False)
                items = pd.concat(
                    [items, item.loc[item["item_code"].isin(item_code)]])
            else:

                # select items based on the top n-percent of the class ratio
                item_top = item.head(int(total_sku*conf))
                items = pd.concat([items, item_top])

        # save items selected to csv
        items.reset_index(drop=True, inplace=True)
        items.index.name = "item_id"

        items_weight = items["box_weight"] / items["number_of_item_in_a_box"]
        items.insert(11, "item_weight", items_weight.round(3))
        items[["item_order_frequency", "number_of_item_in_a_box"]] = items[[
            "item_order_frequency", "number_of_item_in_a_box"]].astype(int)

        item_path = os.path.join(PARENT_DIRECTORY, 'data/output/items.csv')
        items.to_csv(item_path, index=True)

    else:
        items = None

    return items

def gen_pods(pod_num, pod_types, dev_mode=True):

    # get the working path
    working_path = get_working_path(dev_mode)

    # get pod configuration

    pods_dictionary_path = os.path.join(PARENT_DIRECTORY, 'data/input/pods_dictionary.csv')
    pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

    # generate pods based on the pod specification (types and the number of pods)
    counter_id = 0
    pods = pd.DataFrame()
    for i, p in enumerate(pod_types):

        pod_slot_num = pods_dictionary.loc[pods_dictionary["pod_type"] == p].shape[0]
        pod_slot_id = pods_dictionary.loc[pods_dictionary["pod_type"]
                                 == p, "slot_sequence"].to_list()
        slot_type = pods_dictionary.loc[pods_dictionary["pod_type"]
                                 == p, "slot_type"].to_list()       

        pod_type = [p] * pod_slot_num
        pod_face = pods_dictionary.loc[pods_dictionary["pod_type"] == p, "pod_face"].to_list()

        for _ in range(pod_num[i]):
            pod_id = [counter_id] * pod_slot_num
            counter_id += 1

            pod = pd.DataFrame({"pod_id": pod_id,
                                "pod_type": pod_type,
                                "slot_id": pod_slot_id,
                                "slot_type": slot_type,
                                "item": np.nan,
                                "unusedColumn1": 0,
                                "unusedColumn2": 0,
                                "unusedColumn3": 0,
                                "qty": np.nan,
                                "max_qty": np.nan,
                                'due_date': 99999,
                                'facing': pod_face,
                                'pick_ind': 0
                                })
            pods = pd.concat([pods, pod], axis=0)

    pods.reset_index(drop=True, inplace=True)
    return pods

def assign_items_to_pods_single_slot_type(pods, items, items_pods_class_conf, dev_mode=False):

    # For now, this only work if pod only has 1 type of slot.
    # Number of slot must be satisfied with the number of initial invetory of the item.
    # If pod has multiple slot types, then we need to modify the code to consider the item's dimension based on the slot type.

    working_path = get_working_path(dev_mode)
    item_slot_config_path = os.path.join(PARENT_DIRECTORY, 'data/output/items_slots_configuration.csv')
    item_slot_configuration = pd.read_csv(item_slot_config_path, index_col=False)

    # get pod configuration
    pod_path = os.path.join(PARENT_DIRECTORY, 'data/input/pods_dictionary.csv')
    pod_dictionary = pd.read_csv(pod_path, index_col=False)
    pod_types = pods["pod_type"].unique().tolist()
    
    # get the slot types from pod generated (or defined pod types)
    slot_types = pod_dictionary.loc[pod_dictionary["pod_type"].isin(pod_types), "slot_type"].unique().tolist()
    item_slot_configuration = item_slot_configuration.loc[(item_slot_configuration["item_code"].isin(
        items["item_code"])) & (item_slot_configuration["slot_type"].isin(slot_types))]
    
    # find the slot that can store the item
    items.drop("number_of_item_in_a_box", axis=1, inplace=True)
    items = items.merge(item_slot_configuration,
                        how='inner', on='item_code')
    items["number_of_item_in_a_box"] = items["number_of_item_in_a_box"].astype(
        int)
    items["max_box_in_slot"] = items["max_box_in_slot"].astype(int)
    items["item_slot_needed"] = np.ceil(
        (items["item_initial_quantity_inventory"] / items["number_of_item_in_a_box"]) / items["max_box_in_slot"]).astype(int)

    # add item to pod based on the initial quantity inventory
    for item_id in items.index:
        slot_needed = items.loc[item_id, "item_slot_needed"].astype(int)
        pod_available = np.random.choice(
            pods.loc[pods["item"] != item_id, "pod_id"], size=slot_needed, replace=False)

        for pod_id in pod_available:
            slot_available = pods.loc[(pods["pod_id"] == pod_id) & (
                pods["item"].isnull()), "slot_id"]
            if len(slot_available) == 0:
                break
            else:
                slot_id = np.random.choice(slot_available)
                pods.loc[(pods["pod_id"] == pod_id) & (pods["item"].isnull()) & (pods["slot_id"] == slot_id), 
                         "item"] = item_id
                pods.loc[(pods["pod_id"] == pod_id) & (pods["item"] == item_id) & (pods["slot_id"] == slot_id),
                         "qty"] = items.loc[item_id, "number_of_item_in_a_box"] * items.loc[item_id, "max_box_in_slot"]
                pods.loc[(pods["pod_id"] == pod_id) & (pods["item"] == item_id) & (pods["slot_id"] == slot_id),
                         "max_qty"] = items.loc[item_id, "number_of_item_in_a_box"] * items.loc[item_id, "max_box_in_slot"]

    # If there is still slot available, we need to add item to pods. For this case, we will use the class_pod_conf.

    temp = 0
    # calculate cummulative the class_pod_conf value, to get the threshold to assign the item to the pod.
    for key, value in items_pods_class_conf.items():
        if temp == 0:
            temp = value
        else:
            temp += value
            items_pods_class_conf[key] = np.round(temp, 1)

    # sort the class_pod_conf based on the value
    items_pods_class_conf = dict(sorted(items_pods_class_conf.items(), key=lambda x: x[1]))

    # get class_pod_conf values into list as thresholds. The keys will be used to get the item based on the class
    thresholds = list(items_pods_class_conf.values())
    keys = list(items_pods_class_conf.keys())

    # get the pod_id that its slot still has no item assigned
    pod_available = pods.loc[pods["item"].isnull(),
                             "pod_id"].unique().astype(int)

    # assign item to the pod
    if len(pod_available) > 0:
        for pod_id in pod_available:

            # get slot_id that still has no item assigned
            pod = pods.loc[pods["pod_id"] == pod_id]
            pod = pod.sort_values(by="slot_id")
            slot_available = pod.loc[pod["item"].isnull(), "slot_id"].unique()

            # generate random number
            rand = np.random.rand(len(slot_available)).round(2)

            # print(pod)
            # print(pod_id, slot_available, rand)
            for no, r in enumerate(rand):

                # get the items that already assigned to the pod. Each pod cannot have the same item
                item_exist = pod.loc[(pod["item"].notnull()),
                                     "item"].unique().astype(int)
                if r <= thresholds[0]:
                    # print(f"  below {thresholds[0]}, then the class is {keys[0]}")
                    item_available = items.loc[(items["item_class"] == keys[0]) & (
                        ~items.index.isin(item_exist)), "item_order_frequency"]

                for i in range(len(thresholds) - 1):
                    if thresholds[i] < r <= thresholds[i + 1]:
                        # print(f"  between {thresholds[i]} and {thresholds[i + 1]}, then the class is {keys[i+1]}")
                        item_available = items.loc[(items["item_class"] == keys[i+1]) & (
                            ~items.index.isin(item_exist)), "item_order_frequency"]
                # print("    ", item_available)
                item_probability = item_available / item_available.sum()
                item_available = item_available.index.tolist()
                # print("    ", item_available)

                # assign item to the slot
                if len(item_available) > 0:
                    item_id = np.random.choice(
                        item_available, p=item_probability)
                    slot_id = slot_available[no]
                    pods.loc[(pods["pod_id"] == pod_id) & (pods["item"].isnull()) & (
                        pods["slot_id"] == slot_id), "item"] = item_id
                    pods.loc[(pods["pod_id"] == pod_id) & (pods["item"] == item_id) & (pods["slot_id"] == slot_id),
                             "qty"] = items.loc[item_id, "number_of_item_in_a_box"] * items.loc[item_id, "max_box_in_slot"]
                    pods.loc[(pods["pod_id"] == pod_id) & (pods["item"] == item_id) & (pods["slot_id"] == slot_id),
                             "max_qty"] = items.loc[item_id, "number_of_item_in_a_box"] * items.loc[item_id, "max_box_in_slot"]
                else:
                    break

    # make sure the qty and max_qty are integer
    pods[["item", "qty", "max_qty"]] = pods[["item", "qty", "max_qty"]].astype(int)

    # save the pods to csv
    pod_path = os.path.join(PARENT_DIRECTORY, 'data/output/pods.csv')
    pods.to_csv(pod_path, index=False)

    return pods

def assign_items_to_pods(pods, items, items_pods_class_conf, dev_mode=False):

    working_path = get_working_path(dev_mode)
    pods_dictionary_path = os.path.join(PARENT_DIRECTORY, 'data/input/pods_dictionary.csv')
    pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

    # get the slot types from pod generated (or defined pod types)
    slot_types = pods_dictionary.loc[pods_dictionary["pod_type"].isin(pods.loc[:, "pod_type"]), 
                                    ["slot_type", "slot_volume"]].sort_values(by="slot_volume", ascending=False).loc[:, "slot_type"].unique()
    slot_volumes = pods_dictionary.loc[pods_dictionary["pod_type"].isin(pods.loc[:, "pod_type"]), 
                                    ["slot_type", "slot_volume"]].sort_values(by="slot_volume", ascending=False).loc[:, "slot_volume"].unique() 
    pods["slot_sequence"] = np.arange(0, pods.shape[0]) 

    # get the item_slot_configuration based on the items and slot types    
    item_slot_configuration_path = os.path.join(PARENT_DIRECTORY, 'data/output/items_slots_configuration.csv')
    items_slots_configuration = pd.read_csv(item_slot_configuration_path, index_col=False)    
    items_slots_configuration_selected = items_slots_configuration.loc[(items_slots_configuration["item_code"].isin(items["item_code"].unique())) & 
                                                                   (items_slots_configuration["slot_type"].isin(slot_types)) & 
                                                                   (items_slots_configuration["max_box_in_slot"] > 0)]
    
    # merge the item_slot_configuration with the items to get the item_slot_needed and know each item_id's slot_type
    items["item_id"] = items.index
    items_slots_configuration_selected = items_slots_configuration_selected.merge(items[["item_id", 
                                                                                         "item_code", 
                                                                                         "item_class",                                                                                          
                                                                                         "box_weight", 
                                                                                         "item_weight",
                                                                                         "item_order_frequency", 
                                                                                         "item_initial_quantity_inventory", 
                                                                                         "item_pod_inventory_level",
                                                                                         "item_warehouse_inventory_level"]
                                                                                         ].copy(), how='inner', on='item_code')
    items_slots_configuration_selected["item_slot_needed"] = np.ceil(
        (items_slots_configuration_selected["item_initial_quantity_inventory"] / items_slots_configuration_selected["number_of_item_in_a_box"]) / items_slots_configuration_selected["max_box_in_slot"]).astype(int)

    # assign item to the pod based on the slot volume
    for s in range(len(slot_volumes)):
        
        # Filtering the item_slot_configuration based on the slot volume
        if s == (len(slot_volumes)-1):
            # the biggest slot
            items_slots_volume_filtered = items_slots_configuration_selected.loc[(items_slots_configuration_selected["box_volume"] < slot_volumes[s]), 
                                                       ["item_id", "item_class", "slot_type",                                                          
                                                        "number_of_item_in_a_box", "max_box_in_slot", "max_item_in_slot", "item_weight", 
                                                        "item_initial_quantity_inventory", "item_pod_inventory_level", "item_warehouse_inventory_level"
                                                        ]].sort_values(by=["item_class", "item_initial_quantity_inventory"], ascending=[True, False])

            # print(s, slot_volumes[s], len(a))
        else:
            items_slots_volume_filtered = items_slots_configuration_selected.loc[(items_slots_configuration_selected["box_volume"] < slot_volumes[s]) & 
                                                       (items_slots_configuration_selected["box_volume"] >= slot_volumes[s+1]), 
                                                       ["item_id", "item_class", "slot_type",                                                          
                                                        "number_of_item_in_a_box", "max_box_in_slot", "max_item_in_slot", "item_weight", 
                                                        "item_initial_quantity_inventory", "item_pod_inventory_level", "item_warehouse_inventory_level"
                                                        ]].sort_values(by=["item_class", "item_initial_quantity_inventory"], ascending=[True, False])                                            
            # print(s, slot_volumes[s], slot_volumes[s+1], len(a))
        
        if len(items_slots_volume_filtered) > 0:
            item_fitted = items_slots_volume_filtered.loc[:, "item_id"].unique()
            slot_fitted = items_slots_volume_filtered.loc[:, "slot_type"].unique()
            # print("    ", item_fitted, slot_fitted)
    
            for item_id in item_fitted:
                item_initial_quantity_needed = items.loc[items["item_id"] == item_id, "item_initial_quantity_inventory"].values[0]
                while (item_initial_quantity_needed > 0):
                    # print(item_id, item_initial_quantity_needed)
                    pod_available = list(set(pods.loc[(pods["item"] != item_id) & (pods["item"].isnull()) & (pods["slot_type"].isin(slot_fitted)), "pod_id"].unique().tolist()) - 
                                         set(pods.loc[(pods["item"] == item_id) & (pods["item"].notnull()) & (pods["slot_type"].isin(slot_fitted)), "pod_id"].unique().tolist()))    
                    # print("  len(pod_available):", len(pod_available))
                    # print("  pod_available:", pod_available)
                    if len(pod_available) == 0:
                        print("    All pod is unavailable for item_id:", item_id, item_initial_quantity_needed)
                        print("    Break the loop")
                        break
                    else:
                        pod_id = np.random.choice(pod_available)
                        pod = pods.loc[pods["pod_id"] == pod_id]
                        slot_available = pod.loc[(pod["item"] != item_id) & (pod["item"].isnull()) & (pod["slot_type"].isin(slot_fitted)), "slot_id"].unique()
                        # print("    pod_id:", pod_id)
                        # print("    slot_available:", slot_available, len(slot_available))
                        if len(slot_available) == 0:
                            print("      All slot is unavailable in pod_id", pod_id)
                            break
                        else:
                            slot_id = np.random.choice(slot_available)
                            slot_type = pod.loc[pod["slot_id"] == slot_id, "slot_type"].values[0] 

                            max_item_in_slot = items_slots_volume_filtered.loc[(items_slots_volume_filtered["item_id"] == item_id) & 
                                                                               (items_slots_volume_filtered["slot_type"] == slot_type),
                                                                               "max_item_in_slot"].to_numpy()[0]

                            item_weight = items_slots_volume_filtered.loc[(items_slots_volume_filtered["item_id"] == item_id) &
                                                                          (items_slots_volume_filtered["slot_type"] == slot_type), 
                                                                          "item_weight"].to_numpy()[0]
                            
                            qty = max_item_in_slot.astype(int)
                            max_qty = max_item_in_slot.astype(int)
                            total_item_weight = (item_weight * qty).round(3)

                            
                            pods.loc[(pods["pod_id"] == pod_id) & (pods["slot_id"] == slot_id), "item"] = item_id
                            pods.loc[(pods["pod_id"] == pod_id) & (pods["slot_id"] == slot_id), "qty"] = qty
                            pods.loc[(pods["pod_id"] == pod_id) & (pods["slot_id"] == slot_id), "max_qty"] = max_qty
                            pods.loc[(pods["pod_id"] == pod_id) & (pods["slot_id"] == slot_id), "item_weight"] = item_weight
                            pods.loc[(pods["pod_id"] == pod_id) & (pods["slot_id"] == slot_id), "total_item_weight"] = total_item_weight
                            
                            item_initial_quantity_needed -= qty 
                            # print("      ", slot_id, slot_type)
                            # print("      ", number_of_item_in_a_box, max_box_in_slot, qty, max_qty, item_initial_quantity_needed)
    
    # Fill slot that still empty with the item based on the class_pod_conf
    pod_still_available = pods.loc[pods["item"].isnull(), "pod_id"].unique().tolist()

    # calculate the class_pod_conf value, to get the threshold to assign the item to the pod.
    items = pd.merge(items, pd.DataFrame(list(items_pods_class_conf.items()), columns=['item_class', 'item_pod_class_ratio']), on='item_class', how='left')
    
    for pod_id in pod_still_available:
        pod = pods.loc[pods["pod_id"] == pod_id].copy()
        slot_types = pod.loc[pod["item"].isnull(), "slot_type"].unique()
        for slot_type in slot_types:
            
            # get the item that already assigned to the pod. We will exclude this item because a pod cannot has the same item in multiple slot.
            item_exist = pod.loc[(pod["item"].notnull()), "item"].unique().astype(int)
            
            # get the item that can be stored in the slot type
            item_slot_matched = items_slots_configuration_selected.loc[items_slots_configuration_selected["slot_type"]==slot_type, 
                                                                     "item_code"].unique().tolist()
            
            # get the item that still available to be stored in the slot type
            item_available = items.loc[(~items.index.isin(item_exist)) & (items["item_code"].isin(item_slot_matched)), 
                                       ["item_code", "item_pod_class_ratio"]]
            
            # get the item based on the class_pod_conf
            item_probability = item_available["item_pod_class_ratio"] / item_available["item_pod_class_ratio"].sum()
            item_available = item_available.index.tolist()
            
            # get the slot that still empty
            slot_available = pod.loc[(pod["item"].isnull()) & (pod["slot_type"]==slot_type), "slot_id"].unique()
           
            if len(item_available) > 0:

                # select the item based on the class_pod_conf probability
                item_selected = np.random.choice(item_available, size=len(slot_available), p=item_probability, replace=False)
              
                for i, item_id in enumerate(item_selected):

                    # calculate the qty and max_qty based on the item's selected
                    max_item_in_slot = items_slots_configuration_selected.loc[(items_slots_configuration_selected["item_id"]==item_id) &
                                                                              (items_slots_configuration_selected["slot_type"] == slot_type), 
                                                                              "max_item_in_slot"].to_numpy()[0]
                    qty = max_item_in_slot.astype(int)
                    max_qty = max_item_in_slot.astype(int)

                    # calculate the item weight and total item weight
                    item_weight = items_slots_configuration_selected.loc[(items_slots_configuration_selected["item_id"]==item_id) &
                                                                         (items_slots_configuration_selected["slot_type"] == slot_type),
                                                                         "item_weight"].to_numpy()[0]
                    total_item_weight = (item_weight * qty).round(3)                            

                    # assign the item to the slot
                    slot_id = slot_available[i]
                    pod.loc[(pod["pod_id"] == pod_id) & (pod["slot_id"] == slot_id), "item"] = item_id
                    pod.loc[(pod["pod_id"] == pod_id) & (pod["slot_id"] == slot_id), "qty"] = qty
                    pod.loc[(pod["pod_id"] == pod_id) & (pod["slot_id"] == slot_id), "max_qty"] = max_qty
                    pod.loc[(pod["pod_id"] == pod_id) & (pod["slot_id"] == slot_id), "item_weight"] = item_weight
                    pod.loc[(pod["pod_id"] == pod_id) & (pod["slot_id"] == slot_id), "total_item_weight"] = total_item_weight

            else:                
                break

        pods.loc[pods["pod_id"] == pod_id] = pod.loc[pod["pod_id"]==pod_id].copy()
  
    # make sure the qty and max_qty are integer
    pods[["item", "qty", "max_qty"]] = pods[["item", "qty", "max_qty"]].astype(int)

    # save the pods to csv
    pods_path = os.path.join(PARENT_DIRECTORY, 'data/output/pods.csv')
    pods.to_csv(pods_path, index=False)
    
    return pods
                            
def config_items_pods(pod_types=[0], pod_num=[300], total_sku=500, 
                      items_class_conf={"A": 0.1, "B": 0.3, "C": 0.6},
                      items_pods_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6},
                      items_warehouse_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6},
                      items_pods_class_conf={"A": 0.6, "B": 0.3, "C": 0.1},
                      dev_mode=False):
    
    working_path = get_working_path(dev_mode)

    # Configure items and pods based on the pod configuration and the pod configuration.
    config_items_slots(dev_mode=dev_mode)

    # Configure items and pods based on user defined items and pods configuration.
    items_path = os.path.join(PARENT_DIRECTORY, 'data/output/items.csv')
    print("Configuring items...")
    if not os.path.exists(items_path):

        print("    Items configuration is not found. We will generate the items based on the class configuration.")
        items = gen_items(pod_types=pod_types, 
                          total_sku=total_sku, 
                          items_class_conf=items_class_conf,
                          items_pods_inventory_levels=items_pods_inventory_levels,  
                          items_warehouse_inventory_levels=items_warehouse_inventory_levels,
                          select_option=0,
                          dev_mode=dev_mode)       
        items_flag = True
    else:
        items = pd.read_csv(items_path, index_col=0)

        if items.shape[0] == total_sku:
            print("    Items already exist and the number of items is the same as the total SKU.")
            print("    We will use the existing items file.")
            items_flag = False

        elif items.shape[0] > total_sku:
            print("    Items already exist but the number of items is more than the total SKU.")
            print("    We will use the existing items file and select the items based on the total SKU and the class configuration.")
            for k, v in items_class_conf.items():
                items_class = items.loc[items["item_class"]==k]
                items_class = items_class.sort_values(by="item_order_frequency", ascending=False)
                items_class = items_class.head(int(total_sku*v))
                items = pd.concat([items, items_class])
                items = items.drop_duplicates(subset="item_code")
                if items.shape[0] == total_sku:
                    items.reset_index(drop=True, inplace=True)
                    break
            items.to_csv(items_path, index=True)
            items_flag = True
        else:
            print("    Items already exist but the number of items is less than the total SKU.")
            print("    We will generate the items based on number of items needed and the class configuration.")
            items = gen_items(pod_types=pod_types, 
                              total_sku=total_sku, 
                              items_class_conf=items_class_conf,
                              items_pods_inventory_levels=items_pods_inventory_levels,  
                              items_warehouse_inventory_levels=items_warehouse_inventory_levels,
                              select_option=0,
                              dev_mode=dev_mode)   
            items_flag = True
        print("    Items configuration is done. If you want to reconfigure the items, please delete the items.csv file.")
        print()

    if items is not None:
        pods_path = os.path.join(PARENT_DIRECTORY, 'data/output/pods.csv')

        print("Configuring pods and assigning items to the pods...")
        if not os.path.exists(pods_path):

            print("    Pods configuration is not found. We will generate the pods based on the configuration.")
            pods = gen_pods(pod_types=pod_types, pod_num=pod_num, dev_mode=dev_mode)
            pods = assign_items_to_pods(pods, items, items_pods_class_conf=items_pods_class_conf, dev_mode=dev_mode)
            pods_flag = True

        else:
            pods = pd.read_csv(pods_path, index_col=False)
            pod_list = pods["pod_id"].unique().tolist()
            pod_type_list = pods["pod_type"].unique().tolist()

            flag_pod_num_per_type = True
            for i, pod_type in enumerate(pod_types):
                if pod_num[i] != len(pods.loc[pods["pod_type"]==pod_type, "pod_id"].unique().tolist()):
                    flag_pod_num_per_type = False
                    break
            
            flag_pod_type = True
            for pod_type in pod_type_list:
                if pod_type not in pod_types:
                    flag_pod_type = False
                    break
                    
            flag_pod_num = True
            if len(pod_list) != sum(pod_num):
                flag_pod_num = False

            if flag_pod_num_per_type and flag_pod_type and flag_pod_num:
                print("    Pods already exist and the number of pods, number of pod per pods type, and the pod type are the same as the configuration.")
                print("    We will use the existing pods file.")
                pods_flag = False
            else:
                print("    Pods already exist but the number of pods, number of pod per pods type, or the pod type is different from the configuration.")
                print("    We will generate the pods based on the configuration.")
                pods = gen_pods(pod_types=pod_types, pod_num=pod_num, dev_mode=dev_mode)
                pods = assign_items_to_pods(pods, items, items_pods_class_conf=items_pods_class_conf, dev_mode=dev_mode)
                pods_flag = True

        print("    Pods configuration is done. If you want to reconfigure the pods, please delete the pods.csv file.")
        print()
    else:     
        pods = None


    return items, items_flag, pods, pods_flag

if __name__ == "__main__":
    
    dev_mode=True
    pod_types = [0]
    pod_num = [300]
    total_sku = 500
    items_class_conf = {"A": 0.1, "B": 0.3, "C": 0.6}
    items_pod_inventory_levels = {"A": 0.4, "B": 0.5, "C": 0.6}
    items_warehouse_inventory_levels = {"A": 0.4, "B": 0.5, "C": 0.6}
    items_pods_class_conf = {"A": 0.6, "B": 0.3, "C": 0.1}

    working_path = get_working_path(dev_mode)    
    generated_database_order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_database_order.csv')

    # create items and pods (if items and pods are not created yet)
    items, items_flag, pods, pods_flag = config_items_pods(pod_types=pod_types, 
                                                           pod_num=pod_num, 
                                                           total_sku=total_sku, 
                                                           items_class_conf=items_class_conf,
                                                           items_pod_inventory_levels=items_pod_inventory_levels,
                                                           items_warehouse_inventory_levels=items_warehouse_inventory_levels,
                                                           items_pods_class_conf=items_pods_class_conf,
                                                           dev_mode=dev_mode
                                                           )
