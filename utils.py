# -*- coding: utf-8 -*-
"""
Tools for processing DLT log

Created on Wed Jan 29 14:55:38 2020

"""

import os
import shutil
import tarfile
import re
import pandas as pd
import numpy as np
import subprocess
import inspect
import dateutil.parser

class Loader:
    VERBOSE = False

    DLT_VIEWER = 'E:/DLT Viewer/dlt_viewer.exe'
    DLT_VIEWER_ARGS = ['-s', '-u', '-c']
    TEMP_DIR = '.temp'

    DATA_COLS = ['Src', 'SrcIndex', 'Time', 'Timestamp', 'Count', 'Ecuid', 'Apid',
       'Ctid', 'SessionId', 'Type', 'Subtype', 'Mode', 'Args', 'Payload']

    def tar_to_csvgz(tar_filename :str, *, 
                     out_path=".", not_preserve_path=False, ignore_error=False ) -> str :
        try:
            df = Loader.from_tar(tar_filename)
            
            outfile = f"{os.path.splitext(tar_filename)[0]}.csv.gz"
            if not_preserve_path:
                outfile = os.path.split(outfile)[1]
            outfile = f"{out_path}/{outfile}"
            
            os.makedirs(os.path.split(outfile)[0], exist_ok=True)
            Loader.to_csv(df, outfile)
            Loader.printLog(f"{tar_filename} to {outfile}")
                
            Loader.removeTempDir()
            del(df)
            Loader.printLog(f"{Loader.TEMP_DIR} is removed")

        except Exception as e:
            print(e)
            if ignore_error:
                return None
            raise
        return outfile

    def to_csv(df :pd.DataFrame, csv_filename :str):
        if not isinstance(df, pd.DataFrame):
            raise TypeError
        if not isinstance(csv_filename, str):
            raise TypeError

        df = df[Loader.DATA_COLS]
        df.index.name = 'Index'

        df.to_csv(csv_filename)
        return

    def from_csv(csv_filename :str) -> pd.DataFrame :
        if not isinstance(csv_filename, str):
            raise TypeError

        df = pd.read_csv(csv_filename, parse_dates=['Time'], infer_datetime_format=True)
        df = df.set_index('Index')
        df = df[Loader.DATA_COLS]
        return df


    def from_tar(tar_filename :str ) -> pd.DataFrame :
        if not isinstance(tar_filename, str):
            raise TypeError
            
        def filter_tar(members):
            for tarinfo in members:
                if os.path.splitext(tarinfo.name)[1] == '.dlt':
                    yield tarinfo
        
        tar = tarfile.open(tar_filename)
        tar.extractall(Loader.TEMP_DIR, members=filter_tar(tar))
        dlt_list = sorted([f"{Loader.TEMP_DIR}/{f.name}" for f in filter_tar(tar)])
        Loader.printLog("dlt_list:\n{}".format('\n'.join(dlt_list)))
        
        return Loader.from_dlt(dlt_list)
            
    def from_dlt_path( dlt_path :str ) -> pd.DataFrame :
        if not isinstance(dlt_path, str):
            raise TypeError
            
        dlt_list = [f"{dlt_path}/{f}" for f in os.listdir(dlt_path) 
                                    if os.path.splitext(f)[1]==".dlt" ]
        Loader.printLog("dlt_list:\n{}".format('\n'.join(dlt_list)))

        return Loader.from_dlt(dlt_list)
    
    def from_dlt( dlt_list :str or list ) -> pd.DataFrame :
        if isinstance(dlt_list, str):
            dlt_list = [dlt_list]
        if not isinstance(dlt_list, list):
            raise TypeError
            
        text_list = Loader._convertBatch(dlt_list, Loader.TEMP_DIR)
        Loader.printLog("text_list:\n{}".format('\n'.join(text_list)))

        return Loader.from_text(text_list)
    
    def from_text( text_list :list ) -> pd.DataFrame :
        if isinstance(text_list, str):
            text_list = [text_list]
        if not isinstance(text_list, list):
            raise TypeError
            
        raw = pd.Series()
        raw_file = pd.Series()
        for name in text_list:
            with open(name, "r", encoding="utf-8") as f:
                s = pd.Series(f.readlines())
                if s.empty:
                    print(f"Warning!! {name} has no content")
                    continue
                s = s.str.split()
                s_file = pd.Series(np.full_like(s, os.path.split(name)[1]))
                raw = raw.append(s, ignore_index=True)
                raw_file = raw_file.append(s_file, ignore_index=True)

        Loader.printLog("raw.tail()\n{}".format(raw.tail()))

        df = pd.DataFrame()
        df['Src'] =raw_file
        df['SrcIndex'] = raw.map( lambda x: int(x[0]) )
        df['Time'] = raw.map( lambda x: dateutil.parser.parse(" ".join(x[1:3])) )
        df['Timestamp'] = raw.map( lambda x: float(x[3]) )
        df['Count'] = raw.map( lambda x: int(x[4]) )
        df['Ecuid'] = raw.map( lambda x: x[5] )
        df['Apid'] = raw.map( lambda x: x[6] )
        df['Ctid'] = raw.map( lambda x: x[7] )
        df['SessionId'] = raw.map( lambda x: int(x[8]) )
        df['Type'] = raw.map( lambda x: x[9] )
        df['Subtype'] = raw.map( lambda x: x[10] )
        df['Mode'] = raw.map( lambda x: x[11] )
        df['Args'] = raw.map( lambda x: int(x[12]) )
        df['Payload'] = raw.map( lambda x: " ".join(x[13:]) )
        df.index.name = 'Index'

        Loader.printLog("df.columns\n{}".format(df.columns))
        Loader.printLog("df.tail()\n{}".format(df.tail()))
        
        del(raw)
        return df
    
    def _convertToUnicode(dlt_file, out_file, *, ignore_error=False):
        cmd = [Loader.DLT_VIEWER] + Loader.DLT_VIEWER_ARGS + [dlt_file, out_file]
        Loader.printLog(cmd)
            
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            msg = f"_convertToUnicode is error. result={result}"
            if ignore_error:
                print(msg)
                return False
            else:
                raise Loader.DLTError(msg)
        
        Loader.printLog("Success")
            
        return True
    
    def _convertBatch(dlt_list, out_path):
        outputs = []
        os.makedirs(out_path, exist_ok=True)
        
        for dlt_file in dlt_list:
            out_file = f"{out_path}/{Loader.get_valid_filename(dlt_file)}.log"
            Loader._convertToUnicode(dlt_file, out_file)
            outputs.append(out_file)
            
        return outputs
    
    # refer from https://github.com/django/django/blob/master/django/utils/text.py
    def get_valid_filename(s):
        """
        Return the given string converted to a string that can be used for a clean
        filename. Remove leading and trailing spaces; convert other spaces to
        underscores; and remove anything that is not an alphanumeric, dash,
        underscore, or dot.
        >>> get_valid_filename("john's portrait in 2004.jpg")
        'johns_portrait_in_2004.jpg'
        """
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)
    
    def removeTempDir():
        shutil.rmtree(Loader.TEMP_DIR)
        
    def printLog(msg):
        if not Loader.VERBOSE:
            return
        func = inspect.currentframe().f_back.f_code
        print(f"==[{func.co_filename}:{func.co_firstlineno} {func.co_name}] {msg}")
        
    class DLTError(Exception):
        """Exception raised for errors in DLT

        Attributes:
            message -- Some Message that you want to explain
        """
        def __init__(self, message=None):
            self.message = message

        def __str__(self):
            return f"DLTError : {self.message}"


class FilteredData:
    def __init__(self, origin_data :pd.DataFrame, *, prehistory :list=None):
        self.__o_data = origin_data
        self.__f_data = self.__o_data
        self.__history = []

        if prehistory:
            self.apply(prehistory)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'<FilteredData> {len(self.__history)} filter(s) are applied :\n{self.__f_data}'

    @property
    def origin_data(self):
        return self.__o_data

    @property
    def result(self):
        return self.__f_data

    @property
    def history(self):
        return self.__history

    def init_filter(self):
        self.__f_data = self.__o_data
        old_history = self.__history
        self.__history = []
        return old_history

    def apply(self, history :list, *, is_append=False):
        if not isinstance(history, list):
            raise TypeError
        if not is_append:
            self.init_filter()
        for cmd in history:
            cmd[0](self, *cmd[1:])
        return self

    def __val_to_list(self, val):
        return val if isinstance(val, list) else [val]

    def focus(self, col_name:str, val):
        self.__f_data = self.__f_data[self.__f_data[col_name].isin(self.__val_to_list(val))]
        self.__history.append([FilteredData.focus, col_name, val])
        return self

    def exclude(self, col_name:str, val):
        self.__f_data = self.__f_data[~self.__f_data[col_name].isin(self.__val_to_list(val))]
        self.__history.append([FilteredData.exclude, col_name, val])
        return self

    def combine(self, col_name:str, val):
        self.__f_data = self.__f_data.combine_first(
                self.__o_data[self.__o_data[col_name].isin(self.__val_to_list(val))])
        self.__history.append([FilteredData.combine, col_name, val])
        return self
