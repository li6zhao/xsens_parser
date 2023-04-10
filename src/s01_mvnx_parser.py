import numpy as np
from bs4 import BeautifulSoup as bs
import pickle


class MvnxFileParser:
    
    def __init__(self, file_path, parse_full_recording = True, treat_com_as_segment = True):
        '''
        parse the mvnx file
        '''
        with open(file_path) as file:
            self.soup = bs(file, 'xml')
        
        
        # metadata
        soup_mvn = self.soup.find('mvn')
        soup_subject = self.soup.find('subject')
        
        self.metadata = {'mvn_version' : soup_mvn['version'],
                         'mvn_build' : soup_mvn['build'],
                         'torsoColor' : soup_subject['torsoColor'],
                         'frameRate' : int(soup_subject['frameRate']),
                         'segmentCount' : int(soup_subject['segmentCount']),
                         'recDate' : soup_subject['recDate'],
                         'recDateMSecsSinceEpoch' : int(soup_subject['recDateMSecsSinceEpoch']),
                         'configuration' : soup_subject['configuration'],
                         'userScenario' : soup_subject['userScenario'],
                         'processingQuality' : soup_subject['processingQuality'],
                         #'originalFilename' : soup_subject['originalFilename'],
                         'start_frame_global' : None,
                         'end_frame_global' : None,
                         'frame_ind_global' : None
                         }
        
        
        # collects all parsed dataset from recording (each as an empty dictionary)
        self.all_frame_dataset = {'normal' : {},
                                  'identity' :  {},
                                  'tpose' : {},
                                  'tpose-isb' :{}
                           }
        
        self.recording_dataset_lookup = None

        self.dataset_per_frame_type = None
        
        # parse segments
        all_segments_tag = self.soup.find_all('segment')
        self.segments = {segment['label'] : {point['label']:
                                             np.array([float(p) for p in point.get_text().strip().split(' ')] )
                                             for point in segment.findAll('point')}
                         for segment in all_segments_tag}
        
        
        
        # parse sensors
        self.sensors = [sensor['label'] for sensor in  self.soup.find_all('sensor')] 
        
        
        
        
        # #parse joints
        all_joints_tag = self.soup.find_all('joint')
        self.joints = {joint['label']: [connector.get_text() for connector in joint if connector != '\n'] 
                       for joint in all_joints_tag}
        
        
        
        # parse ergo_joints
        all_ergo_joints_tag = self.soup.find_all('ergonomicJointAngle')
        self.ergo_joints = {ergo_joint['label']: [ergo_joint['parentSegment'], ergo_joint['childSegment']] 
                           for ergo_joint in all_ergo_joints_tag}



        #foot_contact_def
        self.foot_contact_def = [foot_contact['label'] for foot_contact in  self.soup.find_all('contactDefinition')]
        
        
        
        self.object_column_groups = {'sensor' : self.sensors,
                                     'segment' : list(self.segments),
                                     'joint' : list(self.joints),
                                     'ergo_joint' : list(self.ergo_joints),
                                     'foot_contact' : list(self.foot_contact_def),
                                     'gnss' : ['Latitude', 'Longitude', 'Altitude'],
                                     'com' : ['position', 'velocity', 'acceleration']
                                     }
        

        if parse_full_recording:
            self._parse_full_recording(treat_com_as_segment)
            self._parse_all_dataset_column_name()

            # delete soup attribute before exporting
            del self.soup
        
        
        
     
        
    def _parse_full_recording(self, adjust_for_com):
        '''
        parse the recorded data (including all 4 frame types)
        '''

        # all recording frames
        self.all_recording_frames = self.soup.find_all('frame', {'type': 'normal'})
        
        if len(self.all_recording_frames) > 0:
            #self.all_dataset_name = [child.name for child in self.all_recording_frames[0].children if child != '\n']
            self.all_dataset_name = {child.name : None for child in self.all_recording_frames[0].children if child != '\n'}
                        
            self.metadata['frame_ms'] = np.array([int(frame['ms']) for frame in self.all_recording_frames] )
            
            self.metadata['total_recording_frames'] = len(self.all_recording_frames)
            
            self._parse_start_end_frame_index()
            

            self._parse_all_frame()
            
            if adjust_for_com:
                self._treat_com_as_segment()
            
            # create a lookup list for all available dataset in each frame type
            self.dataset_per_frame_type = {frame_type : list(self.all_frame_dataset[frame_type]) 
                                           for frame_type in list(self.all_frame_dataset)}

        # clear up the self.all_recording frame after the parsing
        del self.all_recording_frames
        
        

    def _treat_com_as_segment(self):
        '''
        merge the position, velocity and acceleration data of COM to the segment dataset
        '''
        # change dataset
        for ind, var_name in enumerate(['position', 'velocity', 'acceleration']):
            self.all_frame_dataset['normal'][var_name]['data'] = np.concatenate((self.all_frame_dataset['normal'][var_name]['data'], 
                                                                           self.all_frame_dataset['normal']['centerOfMass']['data'][:, ind * 3 : (ind + 1) * 3]),
                                                                          axis=1)          
               
        # change segment defition (treat as a point, not a joint)
        self.segments['COM'] = {'pCOM_Origin': np.array([0., 0., 0.])}
                
        # change all_dataset_name
        del self.all_dataset_name['centerOfMass']
        #self.all_dataset_name.remove('centerOfMass')
                
        # change object_column_groups
        self.object_column_groups['segment'].append('COM')
        
    
    
   
    # future can allow a subset of columns to be parsed (but no only allow for all columns to be parsed)   
    
    def _parse_frame_variable(self, selected_frame, variable_name):
        extracted_array = np.array([np.array([float(p) for p in frame.find(variable_name).get_text().strip().split(' ')])
                                   for frame in selected_frame])
        return extracted_array
    
        
          
    def _parse_frame_data(self, frame_type):
        
        frame_data = self.soup.find_all('frame', {'type' : frame_type})
        
        for variable_name in list(self.all_dataset_name):#self.all_dataset_name:#
            try:
                self.all_frame_dataset[frame_type][variable_name] ={'column_group' : None,
                                                                    'group_axis' : None,
                                                                    'data' : self._parse_frame_variable(frame_data, variable_name)}
            
            except :
                pass

        

    def _parse_all_frame(self):
        for frame_type_name in list(self.all_frame_dataset):
            self._parse_frame_data(frame_type_name)
        

    
    def _parse_start_end_frame_index(self):
        '''
        in an exported recording, the time frames would restart from 0 if only a clip is exported
        this method aligns the frame index to the full recording length
        '''
        
        # get the starting frame's ms
        start_frame_ms = int(self.all_recording_frames[0]['ms'])
        
        frame_ms_diff = start_frame_ms - self.metadata['recDateMSecsSinceEpoch']
        
        if frame_ms_diff % 25 == 0:
            self.metadata['start_frame_global'] = 6 * int(frame_ms_diff / 25)
            self.metadata['end_frame_global'] = self.metadata['start_frame_global'] + (self.metadata['total_recording_frames'] - 1)
        
        else:
            if self.metadata['total_recording_frames'] >= 6:
                # note we always use the first 7 elements, and the initial recording frame's ms is always 0
                first_6_frame_time = np.array([int(frame['time']) for frame in self.all_recording_frames[:6]])
                quotient = (first_6_frame_time + int(self.all_recording_frames[0]['ms'])  - self.metadata['recDateMSecsSinceEpoch']) / 25
                full_cycle_ind = np.where(quotient % 1 == 0)[0][0]
                
                self.metadata['start_frame_global'] = 6 * int(quotient[full_cycle_ind]) - full_cycle_ind
                self.metadata['end_frame_global'] = self.metadata['start_frame_global'] + (self.metadata['total_recording_frames'] - 1)
                
                
 

    def _parse_dataset_column_name(self, frame_type, dataset_name):
        '''
        to help build the mapping between dataset column groups and dataset name
        for later easier data retrieval
        '''
        obj, variable, sub_group = None, None, None
        
        
        # this is used for dataset treating com as a segment
        
        dataset_name_lower = dataset_name.lower()
        
        # note joint angle is a lot more complex than this
        axis_group = ['x', 'y', 'z']
        quat_group = ['q0', 'q1', 'q2', 'q3']

        
        if 'sensor' in dataset_name_lower:
            obj = 'sensor'
        elif 'ergo' in dataset_name_lower:
            obj = 'ergo_joint'
        elif 'joint' in dataset_name_lower:
            obj = 'joint'
        elif dataset_name_lower == 'globalposition':
            obj = 'gnss'
        elif dataset_name_lower == 'footcontacts':
            obj = 'foot_contact'
        else:
            obj = 'segment'
            
        #print(dataset_name_lower)
        if 'orientation' in dataset_name_lower:
            variable = 'orientation'
            sub_group = quat_group
        elif dataset_name_lower == 'globalposition':
            variable = 'gnss'
            sub_group = None
        elif dataset_name_lower == 'footcontacts':
            variable = 'foot_contact'
            sub_group = None
        elif dataset_name_lower == 'angularvelocity':
            variable = 'angular_velocity'
            sub_group = axis_group
        elif dataset_name_lower == 'angularacceleration':
            variable = 'angular_acceleration'
            sub_group = axis_group
        elif dataset_name_lower == 'velocity':
            variable = 'velocity'
            sub_group = axis_group
        elif 'acceleration' in dataset_name_lower:
            variable = 'acceleration'
            sub_group = axis_group
        elif 'jointangle' in dataset_name_lower:
            if 'xzy' in dataset_name_lower:
                variable = 'angle_xzy'
            else:
                variable = 'angle'
            sub_group = axis_group
        elif dataset_name_lower == 'sensormagneticfield':
            variable = 'magnetic_field'
            sub_group = axis_group
        elif dataset_name_lower == 'position':
            variable = 'position'
            sub_group = axis_group
        else:
            variable = 'Unknown'
            sub_group = None
        
        
        # attach the obj_variable to self.all_dataset_name
        self.all_dataset_name[dataset_name] = obj + '_' + variable if sub_group is not None else obj
        
        # attach column names to each dataset 
        column_group = getattr(self, 'object_column_groups')[obj]
        
        
        
        # adjust for COM treatment (as segment)
        if obj == 'segment' and dataset_name not in self.object_column_groups['com']:
            column_group = column_group[:-1]
        
        if sub_group is not None:
            column_group_expanded = ''.join([(grp + '_&_')*len(sub_group) for grp in column_group]).split('_&_')[:-1]
            sub_group_expanded = sub_group * len(column_group)
        else:
            column_group_expanded = column_group
            sub_group_expanded = None

        
        self.all_frame_dataset[frame_type][dataset_name]['column_group'] = np.array(column_group_expanded)
        self.all_frame_dataset[frame_type][dataset_name]['group_axis'] = np.array(sub_group_expanded)
        

    
    def _parse_all_dataset_column_name(self):
        for frame_type in list(self.dataset_per_frame_type):
            for dataset_name in (self.dataset_per_frame_type[frame_type]):
                self._parse_dataset_column_name(frame_type, dataset_name)
        
        # create a dataset lookup list
        self.recording_dataset_lookup = dict(zip(list(self.all_dataset_name.values()), list(self.all_dataset_name.keys())))
    
    
    
    def export(self, file_location):
        # currently only allow to export as pickle
        

        with open(file_location, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
