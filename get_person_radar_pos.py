
def get_person_radar_pos(person_pos_key, radar_pos_key):
    if person_pos_key == "1_0":
        ground_truth = [-3.6, 4.8]
    elif person_pos_key == "1_1":
        ground_truth = [-4.8, 4.8]
    elif person_pos_key == "1_2":
        ground_truth = [-6.0, 4.8]
    elif person_pos_key == "2_0":
        ground_truth = [-3.6, 3.6]
    elif person_pos_key == "2_1":
        ground_truth = [-4.8, 3.6]
    elif person_pos_key == "2_2":
        ground_truth = [-6.0, 3.6]
    else:
        raise Exception("Wrong person position")

    if radar_pos_key == "1-0":
        radar_pos = [-0.6, 2.4]
    elif radar_pos_key == "1-1":
        radar_pos = [-0.6, 1.2]
    elif radar_pos_key == "1-2":
        radar_pos = [-0.6, 0.0]
    elif radar_pos_key == "2-0":
        radar_pos = [0.0, 2.4]
    elif radar_pos_key == "2-1":
        radar_pos = [0.0, 1.2]
    elif radar_pos_key == "2-2":
        radar_pos = [0.0, 0.0]
    else:
        raise Exception("Wrong radar position")
    return ground_truth, radar_pos
