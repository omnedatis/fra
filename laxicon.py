# -*- coding: utf-8 -*-
"""
Created on Sunday 11 13 15:43:02 2022

@author: Jeff
"""
from enum import Enum
import re
from typing import NamedTuple


docstring_pat = '(?<=\n[^#]*)("""[^"][\s\S]*[^"]"""|\'\'\'[^\'][\s\S][^\']*?\'\'\')'


str_literal_pat = '([uU]?[fF]?[bB]?[rR]?[bB]?[fF]?)\'(?!\'\')[\S]*?(?<!\'\')\'|"(?!"")[\S]*?(?<!"")")'


identifier_pat = ''


operator_pat = '=|>=|<=|,|\.|+|-|*|<<|>>|<<=|>>=|**|**=|||||||'


comment_pat = '(?<!["\'])\#.*'


num_literal_pat = '(?<!\w)[1-9][0-9]*\.?[0-9]*'


class Entity(NamedTuple):
    start:str
    pattern:re.Pattern
    end:str='\033[0;37;40m'

class Entities(Entity, Enum):
    # SYMBOL = Entity('\033[0;31;40m')
    LITERRAL = Entity('\033[0;34;40m', re.compile(str_literal_pat, flags=re.VERBOSE))
    OPERATOR = Entity('\033[0;32;40m', re.compile(operator_pat))


def mark(entity:Entity, text:str) -> str:
    ret = ''
    while text:
        match = entity.pattern.search(text)
        if bool(match):
            ret += text[:match.start(0)] + entity.start + match.group(0)+ entity.end
            text = text[match.end(0):]
        else:
            ret += text
            break
    return ret

text = ''
with open('./linked_pinter.py','r') as file:
    for each in file:
        text +=  each

for each in Entities:
    text = mark(each, text)
    print(text)

