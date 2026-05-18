# -*- coding: utf-8 -*-

# Copyright (c) 2015 Chris MacMackin
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Defines an author-year inline citation style for Pybtex. This is 
modified from the alpha style built into Pybtex, written by
Andrey Golovizin.
"""

import re
import sys
import unicodedata
from collections import Counter
from pybtex.style.labels import BaseLabelStyle
from pybtex.textutils import abbreviate

_nonalnum_pattern = re.compile(r'[^\w]+', re.UNICODE)


def _strip_accents(s):
    return u''.join(
        (c for c in unicodedata.normalize('NFD', s)
         if not unicodedata.combining(c)))


def _strip_nonalnum(parts):
    """Strip all non-alphanumerical characters from a list of strings.

    >>> _strip_nonalnum([u"ÅA. B. Testing 12+}[.@~_", u" 3%"])
    u'AABTesting123'
    """
    s = u''.join(parts)
    return _nonalnum_pattern.sub(u'', s)


class LabelStyle(BaseLabelStyle):
    name = 'alpha'

    def format_labels(self, sorted_entries):
        labels = [self.format_label(entry) for entry in sorted_entries]
        for i in range(len(labels)):
            labels[i] = labels[i].replace('\\{', '&#123;')
            labels[i] = labels[i].replace('\\}', '&#125;')
            labels[i] = labels[i].replace('{', '')
            labels[i] = labels[i].replace('}', '')
        count = Counter(labels)
        counted = Counter()
        for label in labels:
            if count[label]:
                yield '(' + label + ')'
            else:
                yield '(' + label + chr(ord('a') + counted[label]) + ')'
                counted.update([label])

    # note: this currently closely follows the alpha.bst code
    # we should eventually refactor it

    def format_label(self, entry):
        # see alpha.bst calc.label
        if entry.type == "book" or entry.type == "inbook":
            label = self.author_editor_key_label(entry)
        elif entry.type == "proceedings":
            label = self.editor_key_organization_label(entry)
        elif entry.type == "manual":
            label = self.author_key_organization_label(entry)
        else:
            label = self.author_key_label(entry)
        if "year" in entry.fields:
            return label.strip() + ', ' + entry.fields["year"]
        else:
            return label.strip()
        # bst additionally sets sort.label

    def author_key_label(self, entry):
        # see alpha.bst author.key.label
        if "author" not in entry.persons:
            if "key" not in entry.fields:
                return entry.key[:]  # entry.key is bst cite$
            else:
                # for entry.key, bst actually uses text.prefix$
                return entry.fields["key"][:]
        else:
            return self.format_lab_names(entry.persons["author"])

    def author_editor_key_label(self, entry):
        # see alpha.bst author.editor.key.label
        if "author" not in entry.persons:
            if "editor" not in entry.persons:
                if "key" not in entry.fields:
                    return entry.key[:]  # entry.key is bst cite$
                else:
                    # for entry.key, bst actually uses text.prefix$
                    return entry.fields["key"][:]
            else:
                return self.format_lab_names(entry.persons["editor"])
        else:
            return self.format_lab_names(entry.persons["author"])

    def author_key_organization_label(self, entry):
        if "author" not in entry.persons:
            if "key" not in entry.fields:
                if "organization" not in entry.fields:
                    return entry.key[:]  # entry.key is bst cite$
                else:
                    result = entry.fields["organization"]
                    if result.startswith("The "):
                        result = result[4:]
                    return result
            else:
                return entry.fields["key"][:]
        else:
            return self.format_lab_names(entry.persons["author"])

    def editor_key_organization_label(self, entry):
        if "editor" not in entry.persons:
            if "key" not in entry.fields:
                if "organization" not in entry.fields:
                    return entry.key[:]  # entry.key is bst cite$
                else:
                    result = entry.fields["organization"]
                    if result.startswith("The "):
                        result = result[4:]
                    return result
            else:
                return entry.fields["key"][:]
        else:
            return self.format_lab_names(entry.persons["editor"])

    @staticmethod
    def format_lab_names(persons):
        # see alpha.bst format.lab.names
        # s = persons
        numnames = len(persons)
        if numnames > 1:
            if numnames > 2:
                namesleft = 1
            else:
                namesleft = numnames
            result = ""
            nameptr = 1
            while namesleft:
                person = persons[nameptr - 1]
                if nameptr == numnames:
                    if str(person) == "others":
                        result += "et al. "
                    else:
                        result += _strip_nonalnum(
                            [abbreviate(name) for name in person.prelast()]
                            + [' ']
                            + person.last()
                        )
                else:
                    result += _strip_nonalnum(
                        [abbreviate(name) for name in person.prelast()]
                        + [' ']
                        + person.last()
                    )
                if numnames == 2 and nameptr == 1:
                    result += ' and '
                else:
                    result += ' '
                nameptr += 1
                namesleft -= 1
            if numnames > 2:
                result += "et al."
        else:
            person = persons[0]
            result = _strip_nonalnum(
                [abbreviate(name) for name in person.prelast()]
                + [' ']
                + person.last()
            )
        return result
