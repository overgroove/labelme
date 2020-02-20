import time
import pandas as pd
import re

class LogTokenizer:
    VERBOSE = False
    
    __positive_id = ['True','Success', 'Good', 'Ok', 'Normal']
    __token_id_positive = (__positive_id
                            + [x.upper() for x in __positive_id]
                            + [x.lower() for x in __positive_id])
    __negative_id = ['False', 'Bad', 'Fatal', 'Exception', 'Exceptions', 'Error', 'Errors',
                     'Crash', 'Warning', 'Critical', 'Assert',
                     'Fail', 'Failure', 'Failed',
                     'Abnormal'
                    ]
    __token_id_negative = (__negative_id
                            + [x.upper() for x in __negative_id]
                            + [x.lower() for x in __negative_id])
    __token_specification = [
        # VALUEs
        ('VAL_EMAIL',  r'[\w\.-]+@[\w\.-]+\.\w{2,4}'),
        ('VAL_IP',     r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}'),
        ('VAL_HEX',    r'(0[Xx])[\wa-fA-F\d]+'),
        ('VAL_FLOAT',  r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?'),
        ('VAL_INT',    r'[+-]?\d+'),
        
        # IDs
        ('ID_CAP',     r'[A-Z]([A-Z\d]?)+(?=([A-Z][a-z\d])|([^A-Za-z\d]|$))'),  # Word in CAPTIAL_CASE_ID
        ('ID_CAMEL',   r'[A-Z][a-z\d]+'),                                       # Word in CamelCaseId
        ('ID_SNAKE',   r'[a-z]([a-z\d]?)+'),                                    # Word in snake_case_id
        ('ID_OTHERS',  r'[A-Za-z\d]+'),                                         # Word that not be in above cases
        
        # SIGNs
        ('SIG_REPEAT', r'(?P<repeat_char>[^A-Za-z\d_])(?P=repeat_char){2,}'), # Repeat special character
        ('SIG_START',  r'[\(\{\[\<]'),                                        # one sign which means block start
        ('SIG_END',    r'[\)\}\]\>]'),                                        # one sign which means block end
        ('SIG_ASSIGN', r'[\=\-\:\;\>]'),                                      # one sign which means assign
        ('SIG_UNDERB', r'_'),
        
        # NO INTERESTs
        ('N_WS',       r'\s'),                                                # whitespace
        ('N_MISMATCH', r'.'),                                                 # Any other character
        
        # Others (Implemented in Function)
        # ('SIG_END_ID',                    r'\s'),
        # ('ID_LOWER',                      r'\s'),
        # ('ID_UPPER',                      r'\s'),
        # ('ID_<something>_POSITIVE',       r'\s'),
        # ('ID_<something>_NEGATIVE',       r'\s'),
    ]
    __token_specification_custom = []
    __case_methods_def = {'lower':['ID_LOWER', str.lower],
                          'upper':['ID_UPPER', str.upper]}

    @property
    def tag_list(self):
        if self.__case_method:
            from_id = [self.__case_method[0]]
        else:
            from_id = [x[0] for x in self.__token_specification if x[0][:3]=='ID_']
        from_id_pos = [f'{x}_POSITIVE' for x in from_id if x[:3]=="ID_"]
        from_id_neg = [f'{x}_NEGATIVE' for x in from_id if x[:3]=="ID_"]
        
        from_val = [x[0] for x in self.__token_specification if x[0][:4]=='VAL_']
        
        from_sig = [x[0] for x in self.__token_specification if x[0][:4]=='SIG_']
        from_sig += ['SIG_END_ID'] if self.__with_sign_end_id else []
        from_sig += ['N_WS', 'N_MISMATCH'] if self.__with_all else []

        from_custom = [x[0] for x in self.__token_specification_custom]
        
        return from_id + from_id_pos + from_id_neg + from_val + from_sig + from_custom
    
    def __init__(self, *, 
                 verbose=False, 
                 batch_size=10000, 
                 handle_case='auto',
                 with_all=False,
                 with_sign_end_id=True
                 ):
        self.VERBOSE = verbose
        self.__batch_size = batch_size
        if handle_case in self.__case_methods_def:
            self.__case_method = self.__case_methods_def[handle_case]
        else:
            self.__case_method = None
        self.__with_all = with_all
        self.__with_sign_end_id = with_sign_end_id
        self.compile_regex()
        return
    
    def __call__(self, sents):
        if isinstance(sents, str):
            return self.tokenize(sents)
        else:
            return self.batch_tokenize(sents)
    
    def compile_regex(self, *, custom_token=[]):
        if custom_token:
            self.__token_specification_custom = custom_token
        tok_regex = '|'.join('(?P<%s>%s)' % pair 
                             for pair in (self.__token_specification_custom +
                                          self.__token_specification))
        self.__compiled_regex = re.compile(tok_regex)
        if self.VERBOSE:
            print(f'tok_regex = {tok_regex}')
        return

    def tokenize(self, s):
        tag_prev = ''
        for mo in self.__compiled_regex.finditer(s):
            tag = mo.lastgroup
            value = mo.group()
    
            if self.__with_sign_end_id:
                # Add ID Seperator for recognizing compound words
                if tag_prev[:3]=='ID_' and tag[:3]!='ID_' and value!='_':
                    yield (' ', 'SIG_END_ID')
            tag_prev = tag
    
            if tag[:3]=="ID_":
                # Deal with lower/upper
                if self.__case_method:
                    tag = self.__case_method[0]
                    value = self.__case_method[1](value)
    
                # Deal with special meaning word
                if value in self.__token_id_positive:
                    tag = f'{tag}_POSITIVE'
                elif value in self.__token_id_negative:
                    tag = f'{tag}_NEGATIVE'
    
            # Ignore unnessary characters
            if not self.__with_all and tag == "N_WS":
                continue
            elif not self.__with_all and tag == 'N_MISMATCH':
                continue
    
            yield (value, tag)
        
        if self.__with_sign_end_id:
            if tag[:3]=='ID_':
                yield (' ', 'SIG_END_ID')
            
        return

    def batch_tokenize(self, sents:pd.Series):
        ss_time = time.time()
        result = pd.Series()
        
        index = 0
        length = sents.shape[0]
        
        while True:
            start, end = index, index + self.__batch_size
            if length <= start:
                break
            if length <= end:
                end = length
                
            s_time = time.time()
            if self.VERBOSE:
                print("==========")
                
            sub_sents = sents.iloc[ start : end ]
            if self.VERBOSE:
                print(sub_sents)
                
            sub_result = sub_sents.apply(lambda x: [(v,k) for v, k in self.tokenize(x)])
            if self.VERBOSE:
                print("----------")
                print(sub_result)
                
            result = result.append(sub_result)
            
            e_time = time.time()
            if self.VERBOSE:
                print("----------")
                print(f'{start}~{end-1}/{length} : batch elapsed {e_time-s_time}, total elapsed {e_time-ss_time}')
            
            del(sub_sents)
            del(sub_result)
            index = end
            
        if self.VERBOSE:
            print("<<<< RESULT >>>>")
            print(result)
            
        return result
