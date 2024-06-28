import plotly.graph_objs as go
import numpy as np
import pickle

from scipy.spatial.transform import Rotation
import math



def crystal_data(d1, d2, h1, h2 = None, h3 = None):
    '''
    to prepare for vertices and faces for crystal shaped mesh
    '''
    
    body_vertices = np.array([[ d1,  0,-h1],
                              [  0, d1,-h1],
                              [-d1,  0,-h1],
                              [  0,-d1,-h1],
                              [ d2,  0, h1],
                              [  0, d2, h1],
                              [-d2,  0, h1],
                              [  0,-d2, h1],
                              
                              # placeholder
                              [  0,  0,  0],
                              [  0,  0,  0]
                         ])
    
    body_lower_ind = np.concatenate([np.arange(4), np.array([0])])
    body_upper_ind = np.concatenate([np.arange(4,8), np.array([4])])
    
    body_faces = [[body_upper_ind[i], body_upper_ind[i+1], body_lower_ind[i]] for i in range(4)]
    body_faces.extend([[body_lower_ind[i], body_lower_ind[i+1], body_upper_ind[i+1]] for i in range(4)])
    
    if h2 is not None:
        
        body_vertices[8] = np.array([ 0, 0, -h2 - h1])
        body_faces.extend([[body_lower_ind[i], body_lower_ind[i+1], 8] for i in range(4)])
    
    if h3 is not None:
        
        body_vertices[9] = np.array([0, 0, h1 + h3])
        body_faces.extend([[body_upper_ind[i], body_upper_ind[i+1], 9] for i in range(4)])
    
    
    return body_vertices, np.array(body_faces)
    

    
    
def hexalid_data(xf_b, xb_b, y1_b, y2_b, y3_b,
                 xf_t, xb_t, y1_t, y2_t, y3_t,
                 h1,
                 cover_top = False,
                 cover_bottom = False):
    
    '''
    to prepare for open lid mesh with hexagon base and top
    '''
    
    # lazy
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9 = xf_b, xb_b, y1_b, y2_b, y3_b, xf_t, xb_t, y1_t, y2_t, y3_t
    
    body_vertices = np.array([[ d0,-d2,  0],
                              [ d0, d2,  0],
                              [  0, d3,  0],
                              [-d1, d4,  0],
                              [-d1,-d4,  0],
                              [  0,-d3,  0],
                              [ d5,-d7, h1],
                              [ d5, d7, h1],
                              [  0, d8, h1],
                              [-d6, d9, h1],
                              [-d6,-d9, h1],
                              [  0,-d8, h1]
                             ])
    
    body_lower_ind = np.concatenate([np.arange(6), np.array([0])])
    body_upper_ind = np.concatenate([np.arange(6,12), np.array([6])])
    
    body_faces = [[body_upper_ind[i], body_upper_ind[i+1], body_lower_ind[i]] for i in range(6)]
    body_faces.extend([[body_lower_ind[i], body_lower_ind[i+1], body_upper_ind[i+1]] for i in range(6)])
    

    if cover_top:
        
        body_faces.extend([[6, body_upper_ind[i+1], body_upper_ind[i+2]] for i in range(4)])
        
    
    if cover_bottom:

        body_faces.extend([[0, body_lower_ind[i+1], body_lower_ind[i+2]] for i in range(4)])
    
    
    return body_vertices, np.array(body_faces)
        
    


def door_stopper_data (x_short, x_long, y, z,
                       seal = False):
    
    '''
    to prepare for "open door-stopper" shaped mesh
    '''
    
    # lazy
    b1, b2, d1, h1 = x_short, x_long, y, z
    
    body_vertices = np.array([[ b2,  0,  0],
                              [-b2,  0,  0],
                              [-b2,  0, h1],
                              [ b2,  0, h1],
                              [ b1,-d1,  0],
                              [-b1,-d1,  0]
                              ])
    
    body_faces = np.array([[1, 4, 0],
                           [1, 4, 5],
                           [2, 4, 3],
                           [2, 4, 5],
                           [0, 3, 4],
                           [1, 2, 5]
                          
                          ])
    
    if seal:
        body_faces = np.concatenate([body_faces,
                                    np.array([[0, 2, 1],
                                              [0, 2, 3]])
                                    ])
    
    return body_vertices, body_faces
    
    
    
    
# I removed sensor mesh - should create it standalone
def get_mesh_data(shape_vertices,
                  shape_faces):
    
    
    shape_mesh = go.Mesh3d(
        x = shape_vertices[:,0],
        y = shape_vertices[:,1],
        z = shape_vertices[:,2],
        colorbar_title='z',
        colorscale='Purples',
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=np.arange(-2, 2, 0.2),
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i = shape_faces[:,0],
        j = shape_faces[:,1],
        k = shape_faces[:,2],
        name='y',
        showscale=True
    )
    
    mesh_data = [shape_mesh]
    
    
    return mesh_data
    



# helper function
def plot_segment_points(mvnx_file_input, segment_name, layout):
    
    segment_points = mvnx_file_input['segments'][segment_name]

    frame_to_plot = np.array(list(segment_points.values())).reshape(-1,3)*100

    # Define the trace for the points
    scatter = go.Scatter3d(
        x=frame_to_plot[:,0],
        y=frame_to_plot[:,1],
        z=frame_to_plot[:,2],
        mode="markers",
        marker=dict(
            size=5,
            color="red"
        ),
        name="Points",
        text = list(segment_points)
    )



    # Create the figure and add the traces and layout
    fig = go.Figure(data=[scatter], layout=layout)

    fig.update_layout(
    #autosize=False,
    width=500,
    height=500,)

    # Show the figure
    fig.show()


class human_mesh:
    def __init__(self, mvnx_file):
        self.mvnx_file = mvnx_file

        frame_to_plot = self.mvnx_file['all_frame_dataset']['tpose']['position']['data'].reshape(-1,3)*100
        all_segments = list(self.mvnx_file['segments'])
        self.all_segments_dict = {i: frame_to_plot[all_segments.index(i)] for i in all_segments[:len(frame_to_plot)]}
        
        self.full_body_scatter =  go.Scatter3d(
                                                x=frame_to_plot[:,0],
                                                y=frame_to_plot[:,1],
                                                z=frame_to_plot[:,2],
                                                mode="markers",
                                                marker=dict(
                                                    size=5,
                                                    color="red"
                                                ),
                                                name="Points",
                                                text = list(mvnx_file['segments'])[:len(frame_to_plot)]
                                            )


        self.full_body_data = {i: None for i in list(self.mvnx_file['segments'])}


    def create_segment_mesh(self):

        # 0. define some useful rotation and other input
        # ------------------------------
        r_y_90 = Rotation.from_euler('y', 90, degrees=True)
        r_x_90 = Rotation.from_euler('x', 90, degrees=True)
        r_x_m_90 = Rotation.from_euler('x', -90, degrees=True)
        r_z_m_90 = Rotation.from_euler('z', -90, degrees=True)
        r_z_90 = Rotation.from_euler('z', 90, degrees=True)

        palm_rotate_deg = 15

        # 1. Pelvis
        # ----------------------------
        pelvis_points =  np.array(list(self.mvnx_file['segments']['Pelvis'].values()))
        pelvis_t_y2 = (self.mvnx_file['segments']['Pelvis']['pLeftCSI'] - self.mvnx_file['segments']['Pelvis']['pRightCSI'])[1] * 100
        pelvis_y3 = (self.mvnx_file['segments']['Pelvis']['pLeftSIPS'] - self.mvnx_file['segments']['Pelvis']['pRightSIPS'])[1] * 100

        pelvis_x_f = (self.mvnx_file['segments']['Pelvis']['pRightASI'] - self.mvnx_file['segments']['Pelvis']['pHipOrigin'])[0] * 100
        pelvis_x_b = (self.mvnx_file['segments']['Pelvis']['pHipOrigin'] - self.mvnx_file['segments']['Pelvis']['pRightSIPS'])[0] * 100

        pelvis_z = (self.mvnx_file['segments']['Pelvis']['jL5S1'] - self.mvnx_file['segments']['Pelvis']['pRightIschialTub'])[2] * 100


        pelvis_origin_offset = np.array([0, 0, min(pelvis_points[:,2])*100])


        pelvis_vertices, pelvis_faces = hexalid_data(xf_t = pelvis_x_f, 
                                                    xb_t = pelvis_x_b,
                                                    y1_t = pelvis_y3/ 2, 
                                                    y2_t = pelvis_t_y2/ 2, 
                                                    y3_t = pelvis_y3/ 2,
                                                    xf_b = pelvis_x_f/ 2, 
                                                    xb_b = pelvis_x_b/ 2, 
                                                    y1_b = pelvis_y3/ 6, 
                                                    y2_b = pelvis_t_y2/ 8, 
                                                    y3_b = pelvis_y3/ 6,
                                                    h1 = pelvis_z,
                                                    cover_top = True,
                                                    cover_bottom = True)

        # apply origin offset
        pelvis_vertices = pelvis_vertices + pelvis_origin_offset


        # append to full body 
        self.full_body_data['Pelvis'] = {'vertices' : pelvis_vertices,
                                    'faces'    : pelvis_faces}
        


        # 2. L5
        # ----------------------------
        l5_points =  np.array(list(self.mvnx_file['segments']['L5'].values()))

        l5_d = (np.max(l5_points[:, 0]) - np.min(l5_points[:, 0])) * 100
        l5_height = (np.max(l5_points[:, 2]) - np.min(l5_points[:, 2])) * 100 / 2

        l5_vertices, l5_faces = crystal_data(d1 = l5_d,    # bottom radius
                                                d2 = l5_d,  # top radius
                                                h1 = l5_height * 0.9,    # body height
                                                h2 = l5_height * 0.1,  # bottom to body height
                                                h3 = l5_height * 0.1     # top to body height
                                                )

        l5_sensor_offset = - l5_vertices[8]

        l5_vertices = l5_vertices + l5_sensor_offset


        # append to full body 
        self.full_body_data['L5'] = {'vertices' : l5_vertices,
                                'faces'    : l5_faces}


        # 3. L3
        # ----------------------------
        l3_points =  np.array(list(self.mvnx_file['segments']['L3'].values()))
        l3_d = (np.max(l3_points[:, 0]) - np.min(l3_points[:, 0])) * 100 
        l3_height = (np.max(l3_points[:, 2]) - np.min(l3_points[:, 2])) * 100 / 2

        l3_vertices, l3_faces = crystal_data(d1 = l3_d,    # bottom radius
                                                d2 = l3_d,  # top radius
                                                h1 = l3_height * 0.9,    # body height
                                                h2 = l3_height * 0.1,  # bottom to body height
                                                h3 = l3_height * 0.1     # top to body height
                                                )

        l3_sensor_offset = - l3_vertices[8]

        l3_vertices = l3_vertices + l3_sensor_offset


        # append to full body 
        self.full_body_data['L3'] = {'vertices' : l3_vertices,
                                'faces'    : l3_faces}


        # 4. T8
        # ----------------------------
        # build a hexagon based/ topped shape

        t8_xb_b_base = - self.mvnx_file['segments']['T8']['pT8SpinalProcess'][0] * 100
        t8_height = self.mvnx_file['segments']['T8']['jT1C7'][2] * 100


        t8_xf_b = t8_xb_b_base * 3
        t8_xb_b = t8_xb_b_base * 1.5
        t8_y1_b = t8_height / 3
        t8_y2_b = t8_height / 3 * 2.5 
        t8_y3_b = t8_height / 3
        t8_xf_t = t8_xb_b_base * 1.5 
        t8_xb_t = t8_xb_b_base * 0.8 
        t8_y1_t = t8_height / 3 
        t8_y2_t = t8_height / 3 * 1.5 
        t8_y3_t = t8_height / 3



        t8_vertices, t8_faces = hexalid_data(xf_b = t8_xf_b, 
                                            xb_b = t8_xb_b, 
                                            y1_b = t8_y1_b, 
                                            y2_b = t8_y2_b, 
                                            y3_b = t8_y3_b,
                                            xf_t = t8_xf_t, 
                                            xb_t = t8_xb_t, 
                                            y1_t = t8_y1_t, 
                                            y2_t = t8_y2_t, 
                                            y3_t = t8_y3_t,
                                            h1 = t8_height,
                                            cover_top = True,
                                            cover_bottom = True)

        # append to full body 
        self.full_body_data_data['T8'] = {'vertices' : t8_vertices,
                                'faces'    : t8_faces}


         # 5. T12
         # ----------------------------
        t12_xb_base = - self.mvnx_file['segments']['T12']['pT12SpinalProcess'][0] * 100
        t12_height = (self.mvnx_file['segments']['T12']['jT9T8'][2] - self.mvnx_file['segments']['T12']['pT12SpinalProcess'][2])  * 100

        # there is some x-direction shift of T12 and T8 origin
        t12_t8_x_offset = (self.all_segments_dict['T8'] - self.all_segments_dict['T12'])[0]


        t12_vertices, t12_faces = hexalid_data(xf_b = t8_xf_t, 
                                            xb_b = t12_xb_base * 0.8, 
                                            y1_b = t8_y1_t, 
                                            y2_b = t8_y2_t, 
                                            y3_b = t8_y3_t,
                                            xf_t = t8_xf_b - abs(t12_t8_x_offset), 
                                            xb_t = t8_xb_b + abs(t12_t8_x_offset), 
                                            y1_t = t8_y1_b, 
                                            y2_t = t8_y2_b, 
                                            y3_t = t8_y3_b,
                                            h1 = t12_height,
                                            cover_top = True,
                                            cover_bottom = True)

        # adjust for top surface widest points
        t12_vertices[8] = np.array([t8_vertices[2][0] + t12_t8_x_offset, t8_vertices[2][1], t12_vertices[8][2]])
        t12_vertices[11] = np.array([t8_vertices[5][0] + t12_t8_x_offset, t8_vertices[5][1], t12_vertices[11][2]])



        # append to full body 
        self.full_body_data['T12'] = {'vertices' : t12_vertices,
                                'faces'    : t12_faces}


        # 6. Neck
        # ----------------------------
        neck_points =  np.array(list(self.mvnx_file['segments']['Neck'].values()))

        neck_d = (np.max(neck_points[:, 0]) - np.min(neck_points[:, 0])) * 100
        neck_height = (np.max(neck_points[:, 2]) - np.min(neck_points[:, 2])) * 100 / 2

        neck_vertices, neck_faces = crystal_data(d1 = neck_d,    # bottom radius
                                                d2 = neck_d,  # top radius
                                                h1 = neck_height
                                                )

        neck_sensor_offset = np.array([0, 0, neck_height])

        neck_vertices = neck_vertices + neck_sensor_offset


        # append to full body 
        self.full_body_data['Neck'] = {'vertices' : neck_vertices,
                                'faces'    : neck_faces}


        # 7. Head
        # ----------------------------
        head_points =  np.array(list(self.mvnx_file['segments']['Head'].values()))

        head_d_t = (self.mvnx_file['segments']['Head']['pLeftAuricularis'] - self.mvnx_file['segments']['Head']['pCenterOfHead'])[1] * 100


        head_d_b = head_d_t * 0.6

        head_height = (np.max(head_points[:, 2]) - np.min(head_points[:, 2])) * 100 / 2

        head_vertices, head_faces = crystal_data(d1 = head_d_b,    # bottom radius
                                                d2 = head_d_t,  # top radius
                                                h1 = head_height * 0.7,    # body height    
                                                h2 = head_height * 0.2,  # bottom to body height
                                                h3 = head_height * 0.4     # top to body height
                                                )

        head_sensor_offset = - head_vertices[8]

        head_vertices = head_vertices + head_sensor_offset



        # append to full body 
        self.full_body_data['Head'] = {'vertices' : head_vertices,
                                'faces'    : head_faces}


        # 8. Left Upper Leg
        # ----------------------------
        left_u_leg_points =  np.array(list(self.mvnx_file['segments']['LeftUpperLeg'].values()))

        left_u_leg_d_b = (self.mvnx_file['segments']['LeftUpperLeg']['jLeftKnee'] - self.mvnx_file['segments']['LeftUpperLeg']['pLeftKneeMedEpicondyle'])[1] * 100

        # need to change Upper leg shape
        left_u_leg_d_t = (self.mvnx_file['segments']['LeftUpperLeg']['pLeftGreaterTrochanter'] - self.mvnx_file['segments']['LeftUpperLeg']['jLeftHip'])[1] * 100 * 0.8

        left_u_leg_height = (np.max(left_u_leg_points[:, 2]) - np.min(left_u_leg_points[:, 2])) * 100 / 2

        left_u_leg_vertices, left_u_leg_faces = crystal_data(d1 = left_u_leg_d_b,    # bottom radius
                                                d2 = left_u_leg_d_t,  # top radius
                                                h1 = left_u_leg_height * 0.9,    # body height
                                                # note, leg height has been divided by 2, and that applies for the body's height
                                                h2 = left_u_leg_height * 0.1,  # bottom to body height
                                                h3 = left_u_leg_height * 0.1     # top to body height
                                                )

        left_u_leg_sensor_offset = left_u_leg_vertices[8]

        left_u_leg_vertices = left_u_leg_vertices + left_u_leg_sensor_offset



        # append to full body 
        self.full_body_data['LeftUpperLeg'] = {'vertices' : left_u_leg_vertices,
                                        'faces'    : left_u_leg_faces}

        # 9. Right Upper Leg
        # ----------------------------
        right_u_leg_points =  np.array(list(self.mvnx_file['segments']['RightUpperLeg'].values()))

        right_u_leg_d_b = (self.mvnx_file['segments']['RightUpperLeg']['jRightKnee'] - self.mvnx_file['segments']['RightUpperLeg']['pRightKneeMedEpicondyle'])[1] * 100

        # need to change Upper leg shape
        right_u_leg_d_t = (self.mvnx_file['segments']['RightUpperLeg']['pRightGreaterTrochanter'] - self.mvnx_file['segments']['RightUpperLeg']['jRightHip'])[1] * 100 * 0.8

        right_u_leg_height = (np.max(right_u_leg_points[:, 2]) - np.min(right_u_leg_points[:, 2])) * 100 / 2

        right_u_leg_vertices, right_u_leg_faces = crystal_data(d1 = right_u_leg_d_b,    # bottom radius
                                                d2 = right_u_leg_d_t,  # top radius
                                                h1 = right_u_leg_height * 0.9,    # body height
                                                # note, leg height has been divided by 2, and that applies for the body's height
                                                h2 = right_u_leg_height * 0.1,  # bottom to body height
                                                h3 = right_u_leg_height * 0.1     # top to body height
                                                )


        right_u_leg_sensor_offset = right_u_leg_vertices[8]

        right_u_leg_vertices = right_u_leg_vertices + right_u_leg_sensor_offset


        # append to full body 
        self.full_body_data['RightUpperLeg'] = {'vertices' : right_u_leg_vertices,
                                        'faces'    : right_u_leg_faces}


        # 10. Left Lower Leg
        # -------------------------------------------
        left_l_leg_points =  np.array(list(self.mvnx_file['segments']['LeftLowerLeg'].values()))

        left_l_leg_d_b = (self.mvnx_file['segments']['LeftLowerLeg']['pLeftLatMalleolus'] - self.mvnx_file['segments']['LeftLowerLeg']['jLeftAnkle'])[1] * 100

        # need to change Upper leg shape
        left_l_leg_d_t = (self.mvnx_file['segments']['LeftLowerLeg']['pLeftTibialTub'] - self.mvnx_file['segments']['LeftLowerLeg']['jLeftKnee'])[0] * 100 * 0.8

        left_l_leg_height = (np.max(left_l_leg_points[:, 2]) - np.min(left_l_leg_points[:, 2])) * 100 / 2

        left_l_leg_vertices, left_l_leg_faces = crystal_data(d1 = left_l_leg_d_b,    # bottom radius
                                                d2 = left_l_leg_d_t,  # top radius
                                                h1 = left_l_leg_height * 0.9,    # body height
                                                h2 = left_l_leg_height * 0.1,  # bottom to body height
                                                h3 = left_l_leg_height * 0.1     # top to body height
                                                )


        left_l_leg_sensor_offset = left_l_leg_vertices[8]

        left_l_leg_vertices = left_l_leg_vertices + left_l_leg_sensor_offset


        # append to full body 
        self.full_body_data['LeftLowerLeg'] = {'vertices' : left_l_leg_vertices,
                                        'faces'    : left_l_leg_faces}
        

        # 11. Right Lower Leg
        # -------------------------------------
        right_l_leg_points =  np.array(list(self.mvnx_file['segments']['RightLowerLeg'].values()))

        right_l_leg_d_b = abs((self.mvnx_file['segments']['RightLowerLeg']['pRightLatMalleolus'] - self.mvnx_file['segments']['RightLowerLeg']['jRightAnkle'])[1]) * 100

        # need to change Upper leg shape
        right_l_leg_d_t = abs((self.mvnx_file['segments']['RightLowerLeg']['pRightTibialTub'] - self.mvnx_file['segments']['RightLowerLeg']['jRightKnee'])[0]) * 100 * 0.8

        right_l_leg_height = (np.max(right_l_leg_points[:, 2]) - np.min(right_l_leg_points[:, 2])) * 100 / 2

        right_l_leg_vertices, right_l_leg_faces = crystal_data(d1 = right_l_leg_d_b,    # bottom radius
                                                d2 = right_l_leg_d_t,  # top radius
                                                h1 = right_l_leg_height * 0.9,    # body height
                                                h2 = right_l_leg_height * 0.1,  # bottom to body height
                                                h3 = right_l_leg_height * 0.1     # top to body height
                                                )



        right_l_leg_sensor_offset = right_l_leg_vertices[8]

        right_l_leg_vertices = right_l_leg_vertices + right_l_leg_sensor_offset


        # append to full body 
        self.full_body_data['RightLowerLeg'] = {'vertices' : right_l_leg_vertices,
                                        'faces'    : right_l_leg_faces}

        # 12. Left Upper Arm
        # ------------------------------
        left_u_arm_points = np.array(list(self.mvnx_file['segments']['LeftUpperArm'].values()))

        left_u_arm_d_b = (self.mvnx_file['segments']['LeftUpperArm']['pLeftArmLatEpicondyle'] - self.mvnx_file['segments']['LeftUpperArm']['jLeftElbow'])[2] * 100 * 0.8

        left_u_arm_d_t = left_u_arm_d_b * 1.2 # random

        left_u_arm_height = (np.max(left_u_arm_points[:, 1]) - np.min(left_u_arm_points[:, 1])) * 100 / 2

        left_u_arm_vertices, left_u_arm_faces = crystal_data(d1 = left_u_arm_d_b,    # bottom radius
                                                d2 = left_u_arm_d_t,  # top radius
                                                h1 = left_u_arm_height * 0.9,    # body height
                                                # note, leg height has been divided by 2, and that applies for the body's height
                                                h2 = left_u_arm_height * 0.1,  # bottom to body height
                                                h3 = left_u_arm_height * 0.1     # top to body height
                                                )

        left_u_arm_sensor_offset = left_u_arm_vertices[8]

        left_u_arm_vertices = left_u_arm_vertices + left_u_arm_sensor_offset

        left_u_arm_vertices = r_x_90.apply(left_u_arm_vertices)



        # append to full body 
        self.full_body_data['LeftUpperArm'] = {'vertices' : left_u_arm_vertices,
                                        'faces'    : left_u_arm_faces}

        # 13. Right Upper Arm
        # -----------------------------

        right_u_arm_points = np.array(list(self.mvnx_file['segments']['RightUpperArm'].values()))

        right_u_arm_d_b = (self.mvnx_file['segments']['RightUpperArm']['pRightArmLatEpicondyle'] - self.mvnx_file['segments']['RightUpperArm']['jRightElbow'])[2] * 100 * 0.8

        right_u_arm_d_t = right_u_arm_d_b * 1.2 # random

        right_u_arm_height = (np.max(right_u_arm_points[:, 1]) - np.min(right_u_arm_points[:, 1])) * 100 / 2

        right_u_arm_vertices, right_u_arm_faces = crystal_data(d1 = right_u_arm_d_b,    # bottom radius
                                                d2 = right_u_arm_d_t,  # top radius
                                                h1 = right_u_arm_height * 0.9,    # body height
                                                # note, leg height has been divided by 2, and that applies for the body's height
                                                h2 = right_u_arm_height * 0.1,  # bottom to body height
                                                h3 = right_u_arm_height * 0.1     # top to body height
                                                )




        right_u_arm_sensor_offset = right_u_arm_vertices[8]

        right_u_arm_vertices = right_u_arm_vertices + right_u_arm_sensor_offset

        right_u_arm_vertices = r_x_m_90.apply(right_u_arm_vertices)


        # append to full body 
        self.full_body_data['RightUpperArm'] = {'vertices' : right_u_arm_vertices,
                                        'faces'    : right_u_arm_faces}

        # 14. Left Forearm
        # ---------------------------------------------
        left_f_arm_points = np.array(list(self.mvnx_file['segments']['LeftForeArm'].values()))

        left_f_arm_d_b = (self.mvnx_file['segments']['LeftForeArm']['pLeftRadialStyloid'] - self.mvnx_file['segments']['LeftForeArm']['jLeftWrist'])[0] * 100 * 0.8

        left_f_arm_d_t = left_f_arm_d_b * 1.2 # random

        left_f_arm_height = (np.max(left_f_arm_points[:, 1]) - np.min(left_f_arm_points[:, 1])) * 100 / 2

        left_f_arm_vertices, left_f_arm_faces = crystal_data(d1 = left_f_arm_d_b,    # bottom radius
                                                d2 = left_f_arm_d_t,  # top radius
                                                h1 = left_f_arm_height * 0.9,    # body height
                                                # note, leg height has been divided by 2, and that applies for the body's height
                                                h2 = left_f_arm_height * 0.1,  # bottom to body height
                                                h3 = left_f_arm_height * 0.1     # top to body height
                                                )


        left_f_arm_sensor_offset = left_f_arm_vertices[8]

        left_f_arm_vertices = left_f_arm_vertices + left_f_arm_sensor_offset

        left_f_arm_vertices = r_x_90.apply(left_f_arm_vertices)


        # append to full body 
        self.full_body_data['LeftForeArm'] = {'vertices' : left_f_arm_vertices,
                                        'faces'    : left_f_arm_faces}

        # 15. Right Forearm
        # -------------------------------------
        right_f_arm_points = np.array(list(self.mvnx_file['segments']['RightForeArm'].values()))

        right_f_arm_d_b = (self.mvnx_file['segments']['RightForeArm']['pRightRadialStyloid'] - self.mvnx_file['segments']['RightForeArm']['jRightWrist'])[0] * 100 * 0.8

        right_f_arm_d_t = right_f_arm_d_b * 1.2 # random

        right_f_arm_height = (np.max(right_f_arm_points[:, 1]) - np.min(right_f_arm_points[:, 1])) * 100 / 2

        right_f_arm_vertices, right_f_arm_faces = crystal_data(d1 = right_f_arm_d_b,    # bottom radius
                                                d2 = right_f_arm_d_t,  # top radius
                                                h1 = right_f_arm_height * 0.9,    # body height
                                                # note, leg height has been divided by 2, and that applies for the body's height
                                                h2 = right_f_arm_height * 0.1,  # bottom to body height
                                                h3 = right_f_arm_height * 0.1     # top to body height
                                                )

        right_f_arm_sensor_offset = right_f_arm_vertices[8]

        right_f_arm_vertices = right_f_arm_vertices + right_f_arm_sensor_offset

        right_f_arm_vertices = r_x_m_90.apply(right_f_arm_vertices)


        # append to full body 
        self.full_body_data['RightForeArm'] = {'vertices' : right_f_arm_vertices,
                                        'faces'    : right_f_arm_faces}

        # 16. Left Shoulder
        # --------------------------------------------------
        left_shoulder_length = abs(self.mvnx_file['segments']['LeftShoulder']['pLeftAcromion'][1] ) * 100
        left_shoulder_height = abs(self.mvnx_file['segments']['LeftShoulder']['pLeftAcromion'][2] ) * 100
        left_shoulder_width = left_f_arm_d_t # left_wrist_width * 0.5 



        left_shoulder_vertices, left_shoulder_faces = door_stopper_data(x_short = left_shoulder_width * 0.8, 
                                                                x_long  = left_shoulder_width, 
                                                                y       = left_shoulder_length, 
                                                                z       = left_shoulder_height,
                                                                    seal = True)


        left_shoulder_sensor_offset = np.array([0 ,left_shoulder_length, 0])

        left_shoulder_vertices = left_shoulder_vertices + left_shoulder_sensor_offset


        # append to full body 
        self.full_body_data['LeftShoulder'] = {'vertices' : left_shoulder_vertices,
                                        'faces'    : left_shoulder_faces}


        # 17. Right Shoulder
        # -------------------------------------------
        right_shoulder_length = abs(self.mvnx_file['segments']['RightShoulder']['pRightAcromion'][1] ) * 100
        right_shoulder_height = abs(self.mvnx_file['segments']['RightShoulder']['pRightAcromion'][2] ) * 100
        right_shoulder_width = right_f_arm_d_t # right_wrist_width * 0.5 



        right_shoulder_vertices, right_shoulder_faces = door_stopper_data(x_short = right_shoulder_width * 0.8, 
                                                                x_long  = right_shoulder_width, 
                                                                y       = - right_shoulder_length, 
                                                                z       = right_shoulder_height,
                                                                        seal = True)

        right_shoulder_sensor_offset = np.array([0 , - right_shoulder_length, 0])

        right_shoulder_vertices = right_shoulder_vertices + right_shoulder_sensor_offset

        # append to full body 
        self.full_body_data['RightShoulder'] = {'vertices' : right_shoulder_vertices,
                                        'faces'    : right_shoulder_faces}



        # 18. Left Foot
        # -------------------------------------------------
        left_ankle_top = (self.mvnx_file['segments']['LeftFoot']['jLeftBallFoot'])[0] * 100# + self.mvnx_file['segments']['LeftToe']['pLeftToe'])[0] * 100

        left_ankle_heel = -self.mvnx_file['segments']['LeftFoot']['pLeftHeelFoot'][0] * 100

        left_foot_height = (self.mvnx_file['segments']['LeftFoot']['jLeftAnkle'] - self.mvnx_file['segments']['LeftFoot']['pLeftHeelCenter'])[2] * 100

        left_toe_width = abs(self.mvnx_file['segments']['LeftFoot']['pLeftFirstMetatarsal'] - self.mvnx_file['segments']['LeftFoot']['pLeftFifthMetatarsal'])[1] * 100 * 0.5

        left_ankle_width = left_toe_width * 0.3

        left_heel_width = left_toe_width * 0.6


        left_foot_vertices, left_foot_faces = door_stopper_data(x_short = left_toe_width, 
                                                                x_long  = left_ankle_width, 
                                                                y       = left_ankle_top, 
                                                                z       = left_foot_height)

        left_heel_points = np.array([[ left_heel_width, left_ankle_heel, 0],
                                    [-left_heel_width, left_ankle_heel, 0]]
                                    )

        left_foot_vertices = np.concatenate([left_foot_vertices, left_heel_points], axis = 0)

        left_heel_faces = left_foot_faces.copy()
        left_heel_faces[np.where(left_heel_faces == 4)] = 6
        left_heel_faces[np.where(left_heel_faces == 5)] = 7

        left_foot_faces = np.concatenate([left_foot_faces, left_heel_faces], axis = 0)

        # rotate
        left_foot_vertices = r_z_90.apply(left_foot_vertices)


        left_foot_sensor_offset = np.array([0, 0, -left_foot_height])

        left_foot_vertices = left_foot_vertices + left_foot_sensor_offset



        # append to full body 
        self.full_body_data['LeftFoot'] = {'vertices' : left_foot_vertices,
                                    'faces'    : left_foot_faces}

        # 19. Right Toe
        # ----------------------------------------
        right_ankle_top = (self.mvnx_file['segments']['RightFoot']['jRightBallFoot'])[0] * 100

        right_ankle_heel = -self.mvnx_file['segments']['RightFoot']['pRightHeelFoot'][0] * 100

        right_foot_height = (self.mvnx_file['segments']['RightFoot']['jRightAnkle'] - self.mvnx_file['segments']['RightFoot']['pRightHeelCenter'])[2] * 100

        right_toe_width = abs(self.mvnx_file['segments']['RightFoot']['pRightFirstMetatarsal'] - self.mvnx_file['segments']['RightFoot']['pRightFifthMetatarsal'])[1] * 100 * 0.5

        right_ankle_width = right_toe_width * 0.3

        right_heel_width = right_toe_width * 0.6


        right_foot_vertices, right_foot_faces = door_stopper_data(x_short = right_toe_width, 
                                                                x_long  = right_ankle_width, 
                                                                y       = right_ankle_top, 
                                                                z       = right_foot_height)

        right_heel_points = np.array([[ right_heel_width, right_ankle_heel, 0],
                                    [-right_heel_width, right_ankle_heel, 0]]
                                    )

        right_foot_vertices = np.concatenate([right_foot_vertices, right_heel_points], axis = 0)

        right_heel_faces = right_foot_faces.copy()
        right_heel_faces[np.where(right_heel_faces == 4)] = 6
        right_heel_faces[np.where(right_heel_faces == 5)] = 7

        right_foot_faces = np.concatenate([right_foot_faces, right_heel_faces], axis = 0)

        # rotate
        right_foot_vertices = r_z_90.apply(right_foot_vertices)


        right_foot_sensor_offset = np.array([0, 0, -right_foot_height])

        right_foot_vertices = right_foot_vertices + right_foot_sensor_offset



        # append to full body 
        self.full_body_data['RightFoot'] = {'vertices' : right_foot_vertices,
                                    'faces'    : right_foot_faces}

        # 20. Left Hand
        # ---------------------------------------
        left_wrist_palm = abs(self.mvnx_file['segments']['LeftHand']['pLeftPinky'][1] *100)

        left_finger_palm = abs(self.mvnx_file['segments']['LeftHand']['pLeftTopOfHand'][1] * 100) - left_wrist_palm

        left_wrist_width = abs(self.mvnx_file['segments']['LeftHand']['pLeftHandPalm'] - self.mvnx_file['segments']['LeftHand']['jLeftWrist'])[0] * 100 * 0.5

        left_palm_width = abs(self.mvnx_file['segments']['LeftHand']['pLeftPinky'] - self.mvnx_file['segments']['LeftHand']['jLeftWrist'])[0] * 100 * 0.5

        left_palm_height = left_wrist_width * 0.4

        # want to get the average for controller and top of hand (otherwise, hand might be too wide)
        left_finger_width  = left_palm_width * 0.7

        left_palm_thickness = left_wrist_width * 1.2



        left_hand_vertices, left_hand_faces = door_stopper_data(x_short = left_wrist_width, 
                                                                x_long  = left_palm_width, 
                                                                y       = left_wrist_palm, 
                                                                z       = left_palm_thickness)


        r_x_left_palm = Rotation.from_euler('x', palm_rotate_deg, degrees=True)

        left_hand_vertices = r_x_left_palm.apply(left_hand_vertices)


        left_palm_h1 = np.sin(math.radians(palm_rotate_deg)) * left_wrist_palm

        left_finger_y = np.sqrt(left_finger_palm**2 - left_palm_h1**2)

        left_finger_points = np.array([[ left_finger_width, left_finger_y, -left_palm_h1],
                                    [-left_finger_width, left_finger_y, -left_palm_h1]])

        left_hand_vertices = np.concatenate([left_hand_vertices, left_finger_points], axis = 0)



        left_hand_sensor_offset = - np.array([0, left_hand_vertices[4][1], left_hand_vertices[4][2]])
        left_hand_vertices = left_hand_vertices + left_hand_sensor_offset



        left_finger_faces = left_hand_faces.copy()
        left_finger_faces[np.where(left_finger_faces == 4)] = 6
        left_finger_faces[np.where(left_finger_faces == 5)] = 7

        left_hand_faces = np.concatenate([left_hand_faces, left_finger_faces], axis = 0)





        # append to full body 
        self.full_body_data['LeftHand'] = {'vertices' : left_hand_vertices,
                                    'faces'    : left_hand_faces}


        # 21. Right Hand
        # --------------------------------------------------
        right_wrist_palm = self.mvnx_file['segments']['RightHand']['pRightPinky'][1] *100

        right_finger_palm = self.mvnx_file['segments']['RightHand']['pRightTopOfHand'][1] * 100 - right_wrist_palm

        right_wrist_width = abs(self.mvnx_file['segments']['RightHand']['pRightHandPalm'] - self.mvnx_file['segments']['RightHand']['jRightWrist'])[0] * 100 * 0.5

        right_palm_width = abs(self.mvnx_file['segments']['RightHand']['pRightPinky'] - self.mvnx_file['segments']['RightHand']['jRightWrist'])[0] * 100 * 0.5

        right_palm_height = right_wrist_width * 0.4

        # want to get the average for controller and top of hand (otherwise, hand might be too wide)
        right_finger_width  = right_palm_width * 0.7

        right_palm_thickness = right_wrist_width * 1.2



        right_hand_vertices, right_hand_faces = door_stopper_data(x_short = right_wrist_width, 
                                                                x_long  = right_palm_width, 
                                                                y       = right_wrist_palm, 
                                                                z       = right_palm_thickness)


        r_x_right_palm = Rotation.from_euler('x', - palm_rotate_deg, degrees=True)

        right_hand_vertices = r_x_right_palm.apply(right_hand_vertices)


        right_palm_h1 = abs(np.sin(math.radians(palm_rotate_deg)) * right_wrist_palm)

        right_finger_y = np.sqrt(right_finger_palm**2 - right_palm_h1**2)

        right_finger_points = np.array([[ right_finger_width, - right_finger_y, -right_palm_h1],
                                    [-right_finger_width, - right_finger_y, -right_palm_h1]])

        right_hand_vertices = np.concatenate([right_hand_vertices, right_finger_points], axis = 0)



        right_hand_sensor_offset = - np.array([0, right_hand_vertices[4][1], right_hand_vertices[4][2]])
        right_hand_vertices = right_hand_vertices + right_hand_sensor_offset



        right_finger_faces = right_hand_faces.copy()
        right_finger_faces[np.where(right_finger_faces == 4)] = 6
        right_finger_faces[np.where(right_finger_faces == 5)] = 7

        right_hand_faces = np.concatenate([right_hand_faces, right_finger_faces], axis = 0)



        # append to full body 
        self.full_body_data['RightHand'] = {'vertices' : right_hand_vertices,
                                    'faces'    : right_hand_faces}


        # 22. Left Toe
        # ------------------------------------------
        left_toe_ball = abs(self.mvnx_file['segments']['LeftToe']['pLeftToe'][0] * 100)
        left_toe_z = abs(self.mvnx_file['segments']['LeftToe']['pLeftToe'][2] * 100)

        left_toe_vertices, left_toe_faces = door_stopper_data(x_short = left_toe_width, 
                                                                x_long  = left_toe_width, 
                                                                y       = left_toe_ball, 
                                                                z       = left_toe_z,
                                                            seal = True)

        left_toe_sensor_offset = np.array([0, 0, -left_toe_z])

        left_toe_vertices = left_toe_vertices + left_toe_sensor_offset

        left_toe_vertices = r_z_90.apply(left_toe_vertices)



        # append to full body 
        self.full_body_data['LeftToe'] = {'vertices' : left_toe_vertices,
                                    'faces'    : left_toe_faces}


        # 23. Right Toe
        # -------------------------------------------
        right_toe_ball = abs(self.mvnx_file['segments']['RightToe']['pRightToe'][0] * 100)
        right_toe_z = abs(self.mvnx_file['segments']['RightToe']['pRightToe'][2] * 100)

        right_toe_vertices, right_toe_faces = door_stopper_data(x_short = right_toe_width, 
                                                                x_long  = right_toe_width, 
                                                                y       = right_toe_ball, 
                                                                z       = right_toe_z,
                                                            seal = True)

        right_toe_sensor_offset = np.array([0, 0, -right_toe_z])

        right_toe_vertices = right_toe_vertices + right_toe_sensor_offset

        right_toe_vertices = r_z_90.apply(right_toe_vertices)


        # append to full body 
        self.full_body_data['RightToe'] = {'vertices' : right_toe_vertices,
                                    'faces'    : right_toe_faces}
