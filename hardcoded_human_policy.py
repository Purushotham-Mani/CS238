# Human-meta actions using combination of [0, 4.5, 9]

def fast_stop_before_intersection(state):
    action = 2
    # Careful as you may cross the lane
    if state[0][1] < 0.019:
        action = 1
    # Slow down before the intersection
    if state[0][2] < 0.243:
        action = 0
    # Stay within the lane
    if state[0][1] < 0.018:
        action = 0
    return action

def probe_intersection(harsh=True):
    action = 1
    if not harsh:
        action = 1
    return action

def cross_intersection(state):
    action = 2
    return action

# Human-meta perception

def check_intersection_region_for_probability(state):
    all_clear = True
    comment = ""
    
    # Confirm if ego-vehicle isn't crossing boundaries
    if state[0][1] < 0.015:
        all_clear = False
        comment = "Lane cross"
        print("Ego-vehicle crossed lane boundary: Probing not allowed")
        return all_clear, comment
    
    for i in range(1, len(state)):
        vehicle = state[i]
        if int(vehicle[0]) == 0:
            break
        
        # Non-threatening vehicles
        # Right lane to top lane
        if vehicle[1] > 0.03 and vehicle[6] < -0.1:
            print("Vehicle ", i, " is not threatening: Moving from right lane to top lane")
            continue
        # Top lane to right lane
        if vehicle[2] < -0.03 and vehicle[5] > 0.1:
            print("Vehicle ", i, " is not threatening: Moving from top lane to right lane")
            continue
        # Intersection
        if vehicle[1] > 0.03 and vehicle[5] > 0.1:
            print("Vehicle ", i, " is not threatening: Within intersection")
            continue
        
        # Check radius
        if np.sqrt(vehicle[1]**2 + vehicle[2]**2) < 0.06:
            all_clear = False
            comment = "Vehicle within radius"
            print("Vehicle ", i, " is threatening: Within radius")
            break
        
    return all_clear, comment

def check_intersection_region_for_crossability(state):
    all_clear = True
    comment = ""
    for i in range(1, len(state)):
        vehicle = state[i]
        if int(vehicle[0]) == 0:
            break
        
        # Non-threatening vehicles
        # Right lane to top lane
        if vehicle[1] > 0.03 and vehicle[6] < -0.1:
            print("Vehicle ", i, " is not threatening: Moving from right lane to top lane")
            continue
        # Top lane to right lane
        if vehicle[2] < -0.03 and vehicle[5] > 0.1:
            print("Vehicle ", i, " is not threatening: Moving from top lane to right lane")
            continue
        # Intersection
        if vehicle[1] > 0.03 and vehicle[5] > 0.1:
            print("Vehicle ", i, " is not threatening: Within intersection")
            continue
        
        # When vehicle leaves the intersection
        # Check down lane
        if vehicle[2] > 0.03 and vehicle[2] > 0.08 and vehicle[6] > 0.0:
            print("Vehicle ", i, " is not threatening: Leaving intersection along down lane")
            continue
        # Check right lane
        if vehicle[1] > 0.03 and vehicle[1] > 0.08 and vehicle[5] > 0.0:
            print("Vehicle ", i, " is not threatening: Leaving intersection along right lane")
            continue
        # Check left lane
        if vehicle[1] < -0.03 and vehicle[1] < -0.11 and vehicle[5] < 0.0:
            print("Vehicle ", i, " is not threatening: Leaving intersection along left lane")
            continue
        
        # When vehicle enters the intersection
        # Check radius
        if np.sqrt(vehicle[1]**2 + vehicle[2]**2) < 0.085:
            all_clear = False
            comment = "Vehicle within radius"
            print("Vehicle ", i, " is threatening: Within radius")
            break
        # Check left lane
        if vehicle[1] < -0.03 and vehicle[1] > -0.32:
            all_clear = False
            comment = "Vehicle in left lane"
            print("Vehicle ", i, " is threatening: In left lane")
            break
        # Check top lane
        if vehicle[2] < -0.03 and vehicle[2] > -0.2:
            all_clear = False
            comment = "Vehicle in top lane"
            print("Vehicle ", i, " is threatening: In top lane")
            break
        # Check right lane
        if vehicle[1] > 0.03 and vehicle[1] < 0.2:
            all_clear = False
            comment = "Vehicle in right lane"
            print("Vehicle ", i, " is threatening: In right lane")
            break
        
    return all_clear, comment

# def check_intersection_region_for_emergency_crossability(state):
#     all_clear = True
#     comment = ""


def state_to_action(state, phase, action_count):
    action = 0
    _, x, y, vx, vy, _, _ = state[0]
    # input("Press Esc or Enter to continue...")
    
    if y > 0.1 and phase == 0:
        input("Phase 0: Press Esc or Enter to continue...")
        action = fast_stop_before_intersection(state)
        
    if np.sqrt(vx ** 2 + vy ** 2) < 0.01 and phase == 0:
        phase = 1
    
    if phase == 1 or phase == 2 and action_count < 1:
        all_clear, comment = check_intersection_region_for_probability(state)
        print(f"Check probability: {all_clear}, {comment}")
        # if comment == "Lane cross":
        #     phase = 2
        input("Phase 1: Press Esc or Enter to continue...")
        if all_clear and action_count == 0:
            action = probe_intersection(True)
            action_count += 1
        elif all_clear and action_count == 1:
            action = probe_intersection(False)
            action_count += 1
        phase = 2
    
    if phase == 2:
        all_clear, comment = check_intersection_region_for_crossability(state)
        print(f"Check crossability: {all_clear}, {comment}")
        input("Phase 2: Press Esc or Enter to continue...")
    
    if phase == 2 and all_clear:
        action_count = 0
        phase = 3
        
    if phase == 3:
        input("Phase 3: Press Esc or Enter to continue...")
        action = cross_intersection(state)
        
    # action = env.action_space.sample()
    
    return action, phase, action_count
