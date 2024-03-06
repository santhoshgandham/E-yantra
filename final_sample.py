import matplotlib.pyplot as plt
from heapq import heappop , heappush
import socket
import time
from time import sleep
import signal
import sys
import numpy as np
import cv2
from cv2 import aruco
import math
import csv

ko = 0
ip = "192.168.79.159"


def read_csv(csv_name):
    priority_order = {}

    # open csv file (lat_lon.csv)
    # read "lat_lon.csv" file
    # store csv data in lat_lon dictionary as {id:[lat, lon].....}
    # return lat_lon

    with open(csv_name, newline='', mode='r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            # Check if the row has at least two elements
            if len(row) >= 2:
                priority_order[row[0]] = row[1]

    return priority_order


dict_with_pref = read_csv ( "priority.csv" )


def read_csv_1(csv_name) :
    lat_lon = {}

    # open csv file (lat_lon.csv)
    # read "lat_lon.csv" file
    # store csv data in lat_lon dictionary as {id:[lat, lon].....}
    # return lat_lon

    '''
    ADD YOUR CODE HERE

    '''
    with open ( csv_name , newline='' ) as csvfile :
        spamreader = csv.reader ( csvfile )
        for row in spamreader :
            lat_lon[row[0]] = [row[1] , row[2]]
    return lat_lon


def tracker(ar_id , lat_lon) :
    # find the lat, lon associated with ar_id (aruco id)
    # write these lat, lon to "live_data.csv"

    if str ( ar_id ) in lat_lon :
        with open ( "live_data.csv" , mode="w" ) as csvfile :
            fieldnames = ["lat" , "lon"]
            writer = csv.DictWriter ( csvfile , fieldnames=fieldnames )
            writer.writeheader ( )
            writer.writerow ( {"lat" : lat_lon[str ( ar_id )][0] , "lon" : lat_lon[str ( ar_id )][1]} )
        coordinate = lat_lon[str ( ar_id )]
        # also return coordinate ([lat, lon]) associated with respective ar_id.
        return coordinate
    else :
        print ( f"Error: Aruco ID {ar_id} not found in lat_lon dictionary." )
        return None


def nearest_aruco(image) :
    ArUco_details_dict = {}
    ArUco_corners = {}
    p = [[21 , 339 , 965] , [20 , 434 , 962] , [19 , 544 , 960] , [16 , 889 , 958] , [17 , 786 , 958] ,
         [18 , 662 , 958] , [15 , 1014 , 953] , [23 , 34 , 922] , [14 , 1044 , 864] , [24 , 39 , 845] ,
         [13 , 1045 , 785] , [22 , 39 , 775] , [25 , 182 , 753] , [26 , 304 , 742] , [28 , 643 , 735] ,
         [29 , 895 , 734] , [27 , 431 , 734] , [11 , 1037 , 613] , [9 , 1030 , 543] , [49 , 24 , 529] ,
         [30 , 881 , 518] , [32 , 607 , 512] , [31 , 683 , 511] , [33 , 431 , 506] , [34 , 330 , 503] ,
         [12 , 1033 , 470] , [50 , 29 , 403] , [8 , 1035 , 390] , [36 , 884 , 354] , [37 , 803 , 350] ,
         [38 , 711 , 346] , [35 , 614 , 343] , [39 , 430 , 338] , [40 , 344 , 330] , [41 , 247 , 327] ,
         [42 , 158 , 323] , [51 , 31 , 273] , [10 , 1035 , 250] , [43 , 843 , 172] , [44 , 756 , 161] ,
         [45 , 647 , 157] , [46 , 539 , 150] , [47 , 437 , 148] , [52 , 31 , 147] , [48 , 350 , 147] ,
         [53 , 70 , 52] , [54 , 174 , 28]]

    arucoDict = cv2.aruco.getPredefinedDictionary ( cv2.aruco.DICT_4X4_250 )
    corners , ids , rejected = cv2.aruco.detectMarkers ( image , arucoDict )

    q = []
    for i in range ( 0 , ids.shape[0] ) :
        a = (corners[i][0][0] + corners[i][0][1] + corners[i][0][2] + corners[i][0][3]) / 4
        b = a.astype ( int )
        l = ids[i]
        l = l.tolist ( )
        d = list ( b )
        l.append ( d[0] )
        l.append ( d[1] )
        q.append ( l )
    for i in q :
        if i[0] == 100 :
            p.append ( [i[0] , i[1] , i[2]] )

    aruco_id_list = []
    for i in p :
        aruco_id_list.append ( i[0] )

    # mention aruco id of bot
    bot_aruco_id = 100
    ar_id = None
    if bot_aruco_id in aruco_id_list :
        aruco_id_list.remove ( bot_aruco_id )
        list_aruco_id_without_bot = aruco_id_list


        # finding coordinates of bot_aruco_id
        list_without_bot_coords_alone = []
        for i in p :
            i[1] = i[1]
            i[2] = i[2]
            b = []
            b.append ( i[1] )
            b.append ( i[2] )
            list_without_bot_coords_alone.append ( b )
            if bot_aruco_id in i :
                a = []
                a.append ( i[1] )
                a.append ( i[2] )
                coords_bot = a
                list_without_bot_coords_alone.remove ( b )



        # finding nearest aruco id to bot_aruco_id
        dist_sum = []
        for i in list_without_bot_coords_alone :
            u = (coords_bot[0] - i[0]) * (coords_bot[0] - i[0]) + (coords_bot[1] - i[1]) * (coords_bot[1] - i[1])
            dist_sum.append ( u )

        ar_id = list_aruco_id_without_bot[dist_sum.index ( min ( dist_sum ) )]

        return ar_id

def signal_handler(sig , frame) :

    cleanup ( )
    sys.exit ( 0 )


def cleanup() :
    s.close ( )


def dijkstra(graph , start , end) :
    heap = [(0 , start , [])]
    visited = set ( )

    while heap :
        (cost , current , path) = heappop ( heap )

        if current in visited :
            continue

        visited.add ( current )
        path = path + [current]

        if current == end :
            return path

        for neighbor , weight in graph[current].items ( ) :
            if neighbor not in visited :
                heappush ( heap , (cost + weight , neighbor , path) )

    return []


def main(pos , bot_center) :
    global ko
    global end_node_list
    graph = {
        'a1' : {'b1' : 2 , 'a2' : 1} ,
        'b1' : {'a1' : 2 , 'b2' : 1} ,
        'a2' : {'a1' : 1 , 'b2' : 2 , 'a3' : 1} ,
        'b2' : {'a2' : 2 , 'c2' : 2 , 'b1' : 1 , 'b3' : 1} ,
        'c2' : {'b2' : 2 , 'c3' : 1} ,
        'a3' : {'a2' : 1 , 'b3' : 2 , 'a4' : 1} ,
        'b3' : {'b2' : 1 , 'b4' : 1 , 'c3' : 2 , 'a3' : 2} ,
        'c3' : {'c2' : 1 , 'b3' : 2 , 'c4' : 1} ,
        'a4' : {'a3' : 1 , 'b4' : 2} ,
        'b4' : {'a4' : 2 , 'b3' : 1 , 'c4' : 2} ,
        'c4' : {'b4' : 2 , 'c3' : 1} ,
    }

    start_node = present_node ( bot_center )
    end_node_list , ko = event_node ( dict_with_pref , ko )

    paths = []
    for _ in range ( 3 ) :
        shortest_path = dijkstra ( graph , start_node , end_node_list[pos] )
        if not shortest_path :
            break
        paths.append ( shortest_path )
        # Update the graph to remove the found path for the next iteration
        for i in range ( len ( shortest_path ) - 1 ) :
            graph[shortest_path[i]].pop ( shortest_path[i + 1] , None )

    if not paths :
        print ( "There is no path between the specified nodes." )
    else :
        for tk , path in enumerate ( paths , start=1 ) :
            continue

    return paths


def direc_event_path(cmd , pth , loops_run) :

    if loops_run < len ( enlst ) :

        if enlst[loops_run] == "A" :

            if len ( cmd ) < 1 :

                cmd.append ( 3 )
            else :

                if pth[-2] == "a1" and pth[-3] == "a2" :
                    cmd.append ( 2 )

                elif pth[-2] == "b1" :
                    cmd.append ( 3 )
            return cmd
        elif enlst[loops_run] == "B" :
            if pth[-2] == "b2" and pth[-3] == "b3" :
                cmd.append ( 2 )
            elif pth[-2] == "b2" and pth[-3] == "a2" :
                cmd.append ( 1 )
            elif pth[-2] == "b2" and pth[-3] == "b1" :
                cmd.append ( 3 )

            elif pth[-2] == "c2" :
                cmd.append ( 3 )
            return cmd
        elif enlst[loops_run] == "C" :
            if len ( cmd ) < 1 :
                cmd.append ( 1 )
            else :
                if pth[-2] == "c3" and pth[-3] == "c4" :
                    cmd.append ( 3 )
                elif pth[-2] == "c3" and pth[-3] == "c2" :
                    cmd.append ( 2 )

                elif pth[-2] == "b3" and pth[-3] == "b2" :
                    cmd.append ( 3 )
                elif pth[-2] == "b3" and pth[-3] == "b4" :
                    cmd.append ( 2 )
                elif pth[-2] == "b3" and pth[-3] == "a3" :
                    cmd.append ( 1 )
            return cmd

        elif enlst[loops_run] == "D" :
            if len ( cmd ) < 1 :
                cmd.append ( 1 )
            else :
                if pth[-2] == "b3" and pth[-3] == "b2" :
                    cmd.append ( 2 )
                elif pth[-2] == "b3" and pth[-3] == "b4" :
                    cmd.append ( 3 )
                elif pth[-2] == "b3" and pth[-3] == "c3" :
                    cmd.append ( 1 )

                elif pth[-2] == "a3" and pth[-3] == "a2" :
                    cmd.append ( 3 )
                elif pth[-2] == "a3" and pth[-3] == "a4" :
                    cmd.append ( 2 )
            return cmd

        elif enlst[loops_run] == "E" :
            if pth[-2] == "a4" and pth[-3] == "b4" :
                cmd.append ( 3 )
            elif pth[-2] == "a4" and pth[-3] == "a3" :
                cmd.append ( 1 )

            elif pth[-2] == "c4" and pth[-3] == "b4" :
                cmd.append ( 2 )
            elif pth[-2] == "c4" and pth[-3] == "c3" :
                cmd.append ( 1 )
        return cmd
    else :
        return cmd


def event_node(dict_with_pref , ko) :
    global enlst
    enlst = list ( dict_with_pref.keys ( ) )
    # values1=list(dict_with_pref.values())

    if ko <= 2 * len ( enlst ) - 1 :
        if enlst[ko // 2] == "A" :
            end_node = ["b1" , "a1"]
        elif enlst[ko // 2] == "B" :
            end_node = ["c2" , "b2"]
        elif enlst[ko // 2] == "C" :
            end_node = ["c3" , "b3"]
        elif enlst[ko // 2] == "D" :
            end_node = ["a3" , "b3"]
        elif enlst[ko // 2] == "E" :
            end_node = ["a4" , "c4"]
    elif ko == 2 * len ( enlst ) or ko == 2 * len ( enlst ) + 1 :

        end_node = ["a1" , "a1"]
    ko += 1
    return end_node , ko


def detect_ArUco_details(image) :
    ArUco_corners = {}

    arucoDict = cv2.aruco.getPredefinedDictionary ( cv2.aruco.DICT_4X4_250 )
    corners , ids , _ = cv2.aruco.detectMarkers ( image , arucoDict )

    if ids is not None and len ( ids ) > 0 :
        for i in range ( ids.shape[0] ) :
            ArUco_corners[int ( ids[i][0] )] = corners[i][0]

    return ArUco_corners


def detect_ArUco_details_1(image) :
    ArUco_details_dict = {}
    ArUco_corners = {}

    arucoDict = cv2.aruco.getPredefinedDictionary ( cv2.aruco.DICT_4X4_250 )
    corners , ids , rejected = cv2.aruco.detectMarkers ( image , arucoDict )

    if ids is not None and ids.shape[0] > 0 :
        for i in range ( 0 , ids.shape[0] ) :
            a = (corners[i][0][0] + corners[i][0][1] + corners[i][0][2] + corners[i][0][3]) / 4
            b = []
            b.append ( int ( a[0] ) )
            b.append ( int ( a[1] ) )
            c = (corners[i][0][0] + corners[i][0][1]) / 2
            slope_1 = (a[0] - c[0]) / (a[1] - c[1])
            slope_2 = abs ( slope_1 )
            angle = math.atan ( slope_2 )
            angle_1 = math.degrees ( angle )
            if c[0] > a[0] and c[1] < a[1] :
                angle_1 = -angle_1
            if c[0] > a[0] and c[1] >= a[1] :
                angle_1 = angle_1 - 180
            if c[0] < a[0] and c[1] <= a[1] :
                angle_1 = angle_1
            if c[0] < a[0] and c[1] > a[1] :
                angle_1 = 180 - angle_1
            if c[0] == a[0] and c[1] > a[1] :
                angle_1 = 180
            angle_1 = int ( angle_1 )
            k = [b , angle_1]
            ArUco_details_dict[int ( ids[i][0] )] = k
            h = corners[i]
            h_1 = h.tolist ( )
            np.array ( h_1[0] )
            ArUco_corners[int ( ids[i][0] )] = np.array ( h_1[0] )
    else :
        print ( "No ArUco markers detected in the current frame." )

    return ArUco_details_dict , ArUco_corners


# Open a connection to the default camera (you can change the argument for a different camera)
def detect_ArUco_center(image , aruco_id) :
    arucoDict = cv2.aruco.getPredefinedDictionary ( cv2.aruco.DICT_4X4_250 )
    corners , ids , _ = cv2.aruco.detectMarkers ( image , arucoDict )

    center_coordinates = None

    if ids is not None and len ( ids ) > 0 :
        for i in range ( ids.shape[0] ) :
            if int ( ids[i][0] ) == aruco_id :
                # Calculate the center coordinates of the ArUco marker
                center_coordinates = np.mean ( corners[i][0] , axis=0 )

    return center_coordinates


def present_node(centre_coordinates) :
    if centre_coordinates[0] <= 199 and centre_coordinates[0] >= 20 and centre_coordinates[1] <= 1020 and \
            centre_coordinates[1] >= 750 :
        node = "a1"
    if centre_coordinates[0] <= 137 and centre_coordinates[0] >= 37 and centre_coordinates[1] <= 737 and \
            centre_coordinates[1] >= 637 :
        node = "a2"
    if centre_coordinates[0] <= 134 and centre_coordinates[0] >= 34 and centre_coordinates[1] <= 487 and \
            centre_coordinates[1] >= 387 :
        node = "a3"
    if centre_coordinates[0] <= 160 and centre_coordinates[0] >= 30 and centre_coordinates[1] <= 330 and \
            centre_coordinates[1] >= 230 :
        node = "a4"
    if centre_coordinates[0] <= 573 and centre_coordinates[0] >= 473 and centre_coordinates[1] <= 946 and \
            centre_coordinates[1] >= 846 :
        node = "b1"
    if centre_coordinates[0] <= 573 and centre_coordinates[0] >= 473 and centre_coordinates[1] <= 722 and \
            centre_coordinates[1] >= 622 :
        node = "b2"
    if centre_coordinates[0] <= 590 and centre_coordinates[0] >= 450 and centre_coordinates[1] <= 510 and \
            centre_coordinates[1] >= 370 :
        node = "b3"
    if centre_coordinates[0] <= 577 and centre_coordinates[0] >= 477 and centre_coordinates[1] <= 330 and \
            centre_coordinates[1] >= 230 :
        node = "b4"
    if centre_coordinates[0] <= 1022 and centre_coordinates[0] >= 922 and centre_coordinates[1] <= 726 and \
            centre_coordinates[1] >= 626 :
        node = "c2"
    if centre_coordinates[0] <= 1025 and centre_coordinates[0] >= 925 and centre_coordinates[1] <= 513 and \
            centre_coordinates[1] >= 413 :
        node = "c3"
    if centre_coordinates[0] <= 999 and centre_coordinates[0] >= 949 and centre_coordinates[1] <= 322 and \
            centre_coordinates[1] >= 272 :
        node = "c4"
    return node


def bot_dir(marker_corners) :
    a = np.mean ( marker_corners , axis=0 )
    c = np.mean ( marker_corners[0 :2] , axis=0 )

    slope_1 = (a[0] - c[0]) / (a[1] - c[1])
    slope_2 = abs ( slope_1 )
    angle_rad = math.atan ( slope_2 )
    angle_deg = math.degrees ( angle_rad )

    if c[0] > a[0] and c[1] < a[1] :
        angle_deg = -angle_deg
    elif c[0] > a[0] and c[1] >= a[1] :
        angle_deg -= 180
    elif c[0] < a[0] and c[1] <= a[1] :
        angle_deg = angle_deg
    elif c[0] < a[0] and c[1] > a[1] :
        angle_deg = 180 - angle_deg
    elif c[0] == a[0] and c[1] > a[1] :
        angle_deg = 180

    # Determine direction based on angle ranges
    if -45 <= angle_deg <= 45 :
        return "e"
    elif 45 < angle_deg <= 135 :
        return "n"
    elif -135 <= angle_deg < -45 :
        return "s"
    else :
        return "w"


def alg_call(possible_paths , dir) :
    penalty = {}
    sequence = {}
    initial_dir = dir  # direc(cmd=[1],initial_dir="n")
    paths = possible_paths

    if initial_dir == "n" :
        i_dir = 1
    elif initial_dir == "e" :
        i_dir = 2
    elif initial_dir == "s" :
        i_dir = 3
    elif initial_dir == "w" :
        i_dir = 4
    else :
        exit ( )

    for i in range ( len ( paths ) ) :
        a = 0
        pen = 0
        seq = []
        while a < len ( paths[i] ) - 1 :
            if paths[i][a][1] < paths[i][a + 1][1] :
                req_dir = 1
            elif paths[i][a][1] > paths[i][a + 1][1] :
                req_dir = 3
            elif paths[i][a][0] < paths[i][a + 1][0] :
                req_dir = 2
            elif paths[i][a][0] > paths[i][a + 1][0] :
                req_dir = 4

            if a == 0 :
                dir = i_dir

            if (req_dir - dir) % 4 == 0 :
                seq = seq + [1 , ]
                pen = pen + 5  # straight
            elif (req_dir - dir) % 4 == 3 :
                seq = seq + [2 , ]  # left
                pen = pen + 9
                dir = dir - 1
            elif (req_dir - dir) % 4 == 1 :
                seq = seq + [3 , ]  # right
                pen = pen + 8
                dir = dir + 1
            elif (req_dir - dir) % 4 == 2 :
                seq = seq + ["Make a U-turn" , ]
                pen = pen + 100
                dir = dir + 2
            a = a + 1
        penalty[i] = pen
        sequence[i] = seq

    a = min ( penalty , key=lambda k : penalty[k] )
    cmd = sequence[a]
    pth = paths[a]

    return cmd , pth , penalty[a]


def best_path(bot_center , dir) :
    possible_path_1 = main ( 0 , bot_center )
    possible_path_2 = main ( 1 , bot_center )
    possible_paths = []
    possible_paths.append ( possible_path_1 )
    possible_paths.append ( possible_path_2 )

    pty_list = []
    cmd_list = []
    pth_list = []
    for vk in possible_paths :
        cmd , pth , pty = alg_call ( vk , dir )
        pty_list.append ( pty )
        cmd_list.append ( cmd )
        pth_list.append ( pth )
    if pty_list[0] < pty_list[1] :
        if len ( pth_list[0] ) > 1 :

            if pth_list[0][-2] != end_node_list[1] :
                pth_list[0].append ( end_node_list[1] )
                updated_cmd = direc_event_path ( cmd_list[0] , pth_list[0] , loops_run )
                return updated_cmd , pth_list[0]
            else :
                return cmd , pth_list[0]

        else :

            pth_list[0].append ( end_node_list[1] )
            updated_cmd = direc_event_path ( cmd_list[0] , pth_list[0] , loops_run )
            return updated_cmd , pth_list[0]

    elif pty_list[1] < pty_list[0] :
        if len ( pth_list[1] ) > 1 :
            if pth_list[1][-2] != end_node_list[0] :
                pth_list[1].append ( end_node_list[0] )
                updated_cmd = direc_event_path ( cmd_list[1] , pth_list[1] , loops_run )
                return updated_cmd , pth_list[1]
            else :
                return cmd , pth_list[1]
        else :
            pth_list[1].append ( end_node_list[0] )
            updated_cmd = direc_event_path ( cmd_list[1] , pth_list[1] , loops_run )
            return updated_cmd , pth_list[1]

    else :
        return cmd_list[1] , pth_list[1]


if __name__ == "__main__" :


    with socket.socket ( socket.AF_INET , socket.SOCK_STREAM ) as s :
        cap = cv2.VideoCapture (0)
        cap.set ( cv2.CAP_PROP_FRAME_WIDTH , 1920 )
        cap.set ( cv2.CAP_PROP_FRAME_HEIGHT , 1080 )

        if not cap.isOpened ( ) :
            print ( "Error: Could not open camera." )
            exit ( )
        rst = 0
        while True :

            ret , frame = cap.read ( )

            if not ret :
                print ( "Error: Could not read frame." )
                break
            ArUco_corners = detect_ArUco_details ( frame )
            if ArUco_corners != {} :
                rst += 1

            # Check if all necessary markers are found
            if all ( key in ArUco_corners for key in [4 , 5 , 6 , 7] ) :
                output_size = (1080 , 1080)

                src_pts = np.float32 (
                    [ArUco_corners[5][2] , ArUco_corners[4][3] , ArUco_corners[6][0] , ArUco_corners[7][1]] )
                dst_pts = np.float32 ( [[0 , 0] , [output_size[0] - 1 , 0] , [output_size[0] - 1 , output_size[1] - 1] ,
                                        [0 , output_size[1] - 1]] )

                perspective_matrix = cv2.getPerspectiveTransform ( src_pts , dst_pts )
                transformed_image = cv2.warpPerspective ( frame , perspective_matrix , output_size )
                resized_transformed_image = cv2.resize ( transformed_image , (1200 , 1200) )
                cv2.imshow ( "Transformed Image" , resized_transformed_image)
                cv2.moveWindow ( "Transformed Image" , 0 , 0 )
            key = cv2.waitKey ( 5 )
            if rst == 10 :
                break
        s.setsockopt ( socket.SOL_SOCKET , socket.SO_REUSEADDR , 1 )
        s.bind ( (ip , 8002) )
        s.listen ( )
        conn , addr = s.accept ( )


        with conn :

            loops_run = 0
            while loops_run <  len( dict_with_pref ) + 1  :


                while True :
                    o=0
                    ret , frame = cap.read ( )
                    if not ret :
                        ret , frame = cap.read ( )
                        if not ret :
                            print ( "Error_Error: Could not read frame." )
                            break

                    ArUco_corners = detect_ArUco_details ( frame )

                    # Check if all necessary markers are found
                    if all ( key in ArUco_corners for key in [4 , 5 , 6 , 7] ) :
                        output_size = (1080 , 1080)

                        src_pts = np.float32 (
                            [ArUco_corners[5][2] , ArUco_corners[4][3] , ArUco_corners[6][0] , ArUco_corners[7][1]] )
                        dst_pts = np.float32 (
                            [[0 , 0] , [output_size[0] - 1 , 0] , [output_size[0] - 1 , output_size[1] - 1] ,
                             [0 , output_size[1] - 1]] )

                        perspective_matrix = cv2.getPerspectiveTransform ( src_pts , dst_pts )
                        transformed_image = cv2.warpPerspective ( frame , perspective_matrix , output_size )

                        ArUco_corners1 = detect_ArUco_details ( transformed_image )

                        # Display the transformed image
                        resized_transformed_image = cv2.resize ( transformed_image , (1200 , 1200) )
                        cv2.imshow ( "Transformed Image" , resized_transformed_image)
                        key=cv2.waitKey(1)
                        cv2.moveWindow ( "Transformed Image" , 0 , 0 )

                        # Show angle for ArUco marker with ID 100
                        if 100 in ArUco_corners1 :

                            w = 0
                            o=0
                            time = 0
                            node_empty = 0
                            ArUco_details_dict , ArUco_corners = detect_ArUco_details_1 ( transformed_image )
                            k = nearest_aruco ( transformed_image )
                            if k is not None :
                                lat_lon = read_csv_1 ( "lat_long.csv" )
                                tracker ( k , lat_lon )
                            while True :

                                data = conn.recv ( 1024 )
                                decoded_data = data.decode ( "utf-8" )
                                integer_data = int ( decoded_data )

                                if integer_data == 1 :

                                    if o == 0 :

                                        dir = bot_dir ( ArUco_corners1[100] )
                                        bot_center = detect_ArUco_center ( transformed_image , 100 )
                                        cmd , pth = best_path ( bot_center , dir )
                                        node_empty = 1
                                        o += 1
                                        loops_run += 1


                                    elif w < len ( cmd ) :
                                        k = str ( cmd[w] )
                                        w += 1
                                        conn.sendall ( str.encode ( k ) )




                                    elif w == len ( cmd ) :
                                        break

                                    integer_data = 0

                                while True :
                                    ret , frame = cap.read ( )
                                    if not ret :
                                        ret , frame = cap.read ( )
                                        if not ret :
                                            break

                                    ArUco_corners = detect_ArUco_details ( frame )

                                    # Check if all necessary markers are found
                                    if all ( key in ArUco_corners for key in [4 , 5 , 6 , 7] ) :
                                        output_size = (1080 , 1080)

                                        src_pts = np.float32 (
                                            [ArUco_corners[5][2] , ArUco_corners[4][3] , ArUco_corners[6][0] ,
                                             ArUco_corners[7][1]] )
                                        dst_pts = np.float32 ( [[0 , 0] , [output_size[0] - 1 , 0] ,
                                                                [output_size[0] - 1 , output_size[1] - 1] ,
                                                                [0 , output_size[1] - 1]] )

                                        perspective_matrix = cv2.getPerspectiveTransform ( src_pts , dst_pts )
                                        transformed_image = cv2.warpPerspective ( frame , perspective_matrix ,
                                                                                  output_size )
                                        resized_transformed_image = cv2.resize ( transformed_image , (1200 , 1200) )
                                        cv2.imshow ( "Transformed Image" , resized_transformed_image )
                                        key = cv2.waitKey ( 2 )
                                        cv2.moveWindow ( "Transformed Image" , 0 , 0 )
                                        ArUco_corners1 = detect_ArUco_details ( transformed_image )
                                    centre = detect_ArUco_center ( transformed_image , 100 )
                                    ArUco_details_dict , ArUco_corners = detect_ArUco_details_1 (
                                        transformed_image )
                                    k = nearest_aruco ( transformed_image )
                                    if k is not None :
                                        lat_lon = read_csv_1 ( "lat_long.csv" )
                                        tracker ( k , lat_lon )

                                    if node_empty > 0 :
                                        if "a1" in end_node_list and "b1" in end_node_list :
                                            if centre is not None :
                                                if centre[0] >= 190 and centre[0] <= 310 and centre[1] >= 870 and \
                                                        centre[1] <= 950 :
                                                    conn.sendall ( str.encode ( "100" ) )
                                        elif "b2" in end_node_list and "c2" in end_node_list :
                                            if centre is not None :
                                                if centre[0] >= 715 and centre[0] <= 830 and centre[1] >= 642 and \
                                                        centre[1] <= 702 :
                                                    conn.sendall ( str.encode ( "100" ) )

                                        elif "b3" in end_node_list and "c3" in end_node_list :
                                            if centre is not None :
                                                if centre[0] >= 725 and centre[0] <= 840 and centre[1] >= 425 and \
                                                        centre[1] <= 495 :
                                                    conn.sendall ( str.encode ( "100" ) )
                                        elif "a3" in end_node_list and "b3" in end_node_list :
                                            if centre is not None :
                                                if centre[0] >= 165 and centre[0] <= 290 and centre[1] >= 407 and \
                                                        centre[1] <= 467 :
                                                    conn.sendall ( str.encode ( "100" ) )
                                        elif "a4" in end_node_list and "c4" in end_node_list :
                                            if centre is not None :
                                                if centre[0] >= 190 and centre[0] <= 310 and centre[1] >= 50 and centre[
                                                    1] <= 110 :
                                                    conn.sendall ( str.encode ( "100" ) )


                                    data = conn.recv ( 1024 )
                                    decoded_data = data.decode ( "utf-8" )
                                    integer_data = int ( decoded_data )

                                    if integer_data == 1 :
                                        break
                    sett=0
                    if o>0:
                        while "a1" in end_node_list and "b1" not in end_node_list:
                            ret , frame = cap.read ( )
                            if not ret :
                                ret , frame = cap.read ( )
                                if not ret :
                                    print ( "Error_Error: Could not read frame." )
                                    break

                            ArUco_corners = detect_ArUco_details ( frame )

                            # Check if all necessary markers are found
                            if all ( key in ArUco_corners for key in [4 , 5 , 6 , 7] ) :

                                output_size = (1080 , 1080)

                                src_pts = np.float32 (
                                    [ArUco_corners[5][2] , ArUco_corners[4][3] , ArUco_corners[6][0] ,
                                     ArUco_corners[7][1]] )
                                dst_pts = np.float32 (
                                    [[0 , 0] , [output_size[0] - 1 , 0] , [output_size[0] - 1 , output_size[1] - 1] ,
                                     [0 , output_size[1] - 1]] )

                                perspective_matrix = cv2.getPerspectiveTransform ( src_pts , dst_pts )
                                transformed_image = cv2.warpPerspective ( frame , perspective_matrix , output_size )
                                resized_transformed_image = cv2.resize ( transformed_image , (1200 , 1200) )
                                cv2.imshow ( "Transformed Image" , resized_transformed_image )
                                key = cv2.waitKey ( 1 )
                                cv2.moveWindow ( "Transformed Image" , 0 , 0 )

                            centre=detect_ArUco_center(transformed_image,100)
                            node=None
                            if centre is not None:
                                node=present_node(centre)
                            if loops_run==len( dict_with_pref ) + 1 and node=="a1":
                                data = conn.recv ( 1024 )
                                decoded_data = data.decode ( "utf-8" )
                                integer_data = int ( decoded_data )
                                dir = bot_dir ( ArUco_corners1[100] )
                                if integer_data == 1 and dir == "s" and sett==0:
                                    conn.sendall ( str.encode ( "1" ) )
                                    sett+=1
                                if integer_data == 1 and dir == "w" and sett==0:
                                    conn.sendall ( str.encode ( "2" ) )
                                    sett+=1
                                if "a1" in end_node_list and "b1" not in end_node_list :
                                    if centre is not None :
                                        if centre[0] >= 0 and centre[0] <= 140 and centre[1] >= 970 and centre[1] <= 1080 :
                                            while(1):
                                                conn.sendall ( str.encode ( "100" ) )
                                                continue
                                            break


                    break
            s.close()
    cap.release ( )
    cv2.destroyAllWindows ( )
