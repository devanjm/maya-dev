from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphanums,
    nums,
    Suppress,
    Optional,
    OneOrMore,
    ZeroOrMore,
)
import random
from string import ascii_lowercase

import maya.cmds as cmds
from maya.api import OpenMaya


_matrixParser = None

_node_type_suffix = {
    'multiplyDivide': 'md',
    'plusMinusAverage': 'pma',
    'addDoubleLinear': 'adl',
    'multDoubleLinear': 'mdl',
    'condition': 'cnd',
    'clamp': 'cmp',
    'angleBetween': 'ang',
    'distanceBetween': 'dst',
    'blendTwoAttr': 'bta',
    'eulerToQuat': 'e2q'
}

def mpe(expression, **kwargs):
    global _matrixParser
    if _matrixParser is None:
        _matrixParser = MatrixParser()
    return _matrixParser.eval(expression, **kwargs)


class InvalidModifierException(Exception):
    pass


class MatrixParser(object):
    """
    Class to create maya dag node network for matrix multiplication.
    Usually this only involves one node.  If you enclose parts of the
    expression in parens, it will create a separate mult for that section.

    Add modifiers after the plug for short hand operations to modify the matrix.
    You can have as many modifiers as you want.

    Special case for the 'G' modifier, which must come last.

    You can also pass in numerical matrices as lists of values.
    We support two types: Nested list of tuples (equivalent of str(OpenMaya.MMatrix)),
    and a flat list of 16 floats equivalent of (cmds.xform(node, ws=1, m=1, q=1)).

    This class can simply be used to compute matrix values for you.
    If no assignment operator "=" is used, and all entries are
    suffixed with the "G" modifier, and/or are all
    numerical matrices, the return will just be a matrix.

    If the above is true, but the assignment operator is found,
    it will assign the computed matrix to the object via xform command.

    modifiers:
        'I'/'inverse': inverse
        'T'/'transpose': transpose
        'asMatrix': compose matrix from attribute
        'pickT': pickMatrix use translate only
        'pickR': pickMatrix use rotate only
        'G': get the matrix value and set it instead of connecting.


    example:

        The most basic use case is basically a 'getAttr'
        This will return an OpenMaya.MMatrix for this plug value.

        >>> mpe('pCube1.matrix.G')

        This will create a mult with [0] driverA.worldMatrix -> pickMatrix (using translate only) *
                                     [1] driverB.parentInverseMatrix
        The output will be plugged into pCube1.offsetParentMatrix.

        >>> mpe('pCube1.offsetParentMatrix = driverA.worldMatrix.pickT * driverB.parentInverseMatrix')

        Nesting expressions inside parens will create a separate mult and network for the expression inside parens
        Because there is no assignment operator "=" the network output matrix plug will be returned.

        >>> mpe('(l_shoulder.matrix * l_shoulder.jointOrient.asMatrix.inverse.G) * l_shoulder.translate.asMatrix')

        You can also use kwargs for short hand like dge which get replaced later

        >>> mpe('driven = driver1_local_matrix * driver2_inverse_parent_matrix', \
                    driven='joint1.offsetParentMatrix', \
                    driver1_local_matrix='jointA.matrix', \
                    driver2_inverse_parent_matrix='jointB.parentMatrixInverse', \
                    decompose=True)

        Kwargs can also partially replace parts of the object.modifiers format in the expression

        >>> mpe('pCube1.offsetParentMatrix = driver.pickT * parent.inverse', \
                    driver='l_hand_ctrl', \
                    parent='l_wrist_jnt.parentMatrix')

        You can also pass matrix values in directly as strings in two formats:
        4 tuples of 4 elements (return of str(OpenMaya.MMatrix)
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        list of 16 values  (return of str(cmds.xform))
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        >>> mpe('pCube1 = {}'.format(str(OpenMaya.MMatrix.kIdentity)))

        ...or you can use kwargs normally

        >>> mpe('pCube1 = cubeMatrix', cubeMatrix=OpenMaya.MMatrix.kIdentity)

    tests:

        # Create test objects
        cube = cmds.polyCube(ch=0)[0]
        cmds.select(cl=1)
        joint = cmds.joint()
        loc = cmds.spaceLocator()[0]
        cmds.setAttr('{}.t'.format(joint), -3.39642709,  2.00734031,  1.30734746)
        cmds.setAttr('{}.r'.format(joint), 78.25749062,  202.05106831,  158.02150562)
        cmds.setAttr('{}.t'.format(loc), 3.17723997,  2.55118277,  -5.27557972)
        cmds.setAttr('{}.r'.format(loc), 109.29794935,  178.02335204,    6.15955528)

        # Test 1 - simple get
        mpe('locator1.matrix.G')

        # Test 2 - basic mult
        mpe('locator1.matrix * joint1.matrix.G')

        # Test 3 - basic mult using replace kwargs
        mpe('loc.matrix * joint.matrix', loc=loc, joint=joint)

        # Test 4 - basic mult with modifiers
        mpe('loc.matrix.I * joint.jointOrient.asMatrix.G', loc=loc, joint=joint)

        # Test 5 - assignment
        mpe('pCube1.opm = loc.matrix.I * joint.jointOrient.asMatrix.G', loc=loc, joint=joint)

        # Test 6 - assignment with decompose
        mpe('pCube1 = loc.matrix.I * joint.jointOrient.asMatrix.G', loc=loc, joint=joint, decompose=True)

        # Test 7 - numerical matrix
        mpe('pCube1.opm = loc.matrix.I * [0.8594902410520135, -0.3468813896816851, 0.3754328528896407, 0.0, 0.2646952686147059, -0.3262916654981234, -0.9074525683469357, 0.0, 0.43727901883047204, 0.8793219265541854, -0.18862663961300896, 0.0, -3.39642709, 2.00734031, 1.30734746, 1.0]', loc=loc)

        # Test 8 - numerical matrix kwarg
        mpe('pCube1.opm = loc.matrix.I * mat', loc=loc, mat=cmds.xform(joint, q=1, ws=1, m=1))

        # Test 9 - parens
        mpe('pCube1.opm = loc.matrix.I * (joint.jointOrient.asMatrix.G * pCube1.parentInverseMatrix.G)', loc=loc, joint=joint)

        # Test 10 - parens with modifiers
        mpe('pCube1.opm = loc.matrix.I * (joint.jointOrient.asMatrix.I * pCube1.parentInverseMatrix).inverse.G', loc=loc, joint=joint)

        # Test 11 - nested parens
        mpe('pCube1.opm = loc.matrix.I * ((joint.jointOrient.asMatrix.I * loc.worldMatrix) * pCube1.parentInverseMatrix.G)', loc=loc, joint=joint)


    """
    def __init__(self):
        super(MatrixParser, self).__init__()

        self.getModifier = 'G'
        self.modifiers = {'I': self.inverse,
                          'inverse': self.inverse,
                          'T': self.transpose,
                          'transpose': self.transpose,
                          'asMatrix': self.asMatrix,
                          'pickT': self.pickT,
                          'pickR': self.pickR,
                          self.getModifier: self.get}

        self.parser = None
        self.outputTarget = None
        self.kwargs = {}
        self.matrixDict = {}
        self.parensDict = {}
        self.expression = None
        self.mults = []

        self.buildParser()

    def _printTokens(self, toks):
        print('printTokens')
        print(toks)

    def buildParser(self):

        # PyParsing expression handler
        # We want to include parens in search, but not return them
        parens = [Suppress(i) for i in '()']
        squareParens = [Suppress(i) for i in '[]']

        # The mult operator, again suppressed.  Mult is all we care about for matrices for 99% of operations.
        mult = Suppress('*')
        dot = Literal('.')
        assignment = Suppress("=")

        # A modifier
        mod_I = Word('I')
        mod_inverse = Word('inverse')
        mod_T = Word('T')
        mod_transpose = Word('transpose')
        mod_asMatrix = Word('asMatrix')
        mod_pickT = Word('pickT')
        mod_pickR = Word('pickR')
        mod_G = Word(self.getModifier)
        modWords = mod_I ^ mod_inverse ^ mod_T ^ mod_transpose ^ mod_asMatrix ^ mod_pickT ^ mod_pickR ^ mod_G
        mods = OneOrMore(dot + modWords)

        # The Word object to represent a node.attr string in all it's combinations
        nodeAttr = OneOrMore(Word(alphanums + '[]_.:'))

        # Looking for the "=" variable assignment
        assignment_op = nodeAttr + assignment

        # Instructions if a numerical matrix value is given.
        # We support two types, nested list of tuples (equivalent of str(OpenMaya.MMatrix))
        # and a flat list of 16 floats, equivalent of cmds.xform(node, ws=1, m=1, q=1)
        comma = Optional(Suppress(','))
        element = OneOrMore(Word(nums + '-.'))
        row = parens[0] + (element + comma) * 4 + parens[1] + comma
        tupleMat = parens[0] * 2 + row * 4 + parens[1] * 2

        # The float 16 list version
        listMat = squareParens[0] + (element + comma) * 16 + squareParens[1]

        # Matrices are replaced with a unique key string and stored in a var
        mat = tupleMat | listMat
        mat.setParseAction(self.storeMatrix)

        # This placeholder for recursive functionality
        terms = Forward()
        # Expression to represent terms inside parenthesis.  When we find those we store the
        # expression inside the parens in a var and replace it with a place holder string
        parenthetical = (parens[0] + Group(terms) + parens[1]).setParseAction(self.storeParens)
        # Term is a matrix, node.attr, or parens + any modifiers
        term = (mat | nodeAttr | parenthetical) + ZeroOrMore(mods)
        terms <<= (term + ZeroOrMore((mult | dot) + terms)).setParseAction(self.joinDots)

        # Possible assignment for connection at the beginning
        assignment = Optional(assignment_op).setParseAction(self.storeOutput) + terms

        self.parser = assignment

    def eval(self, expression, **kwargs):
        """
        Evaluate the expression.  Will convert to dg nodes.

        All kwargs keys are replaced in the input string with the value of the kwarg.
        IE: eval('a * b.inverse', a='joint1', b='joint2') <-- a will be replaced with "joint1"
                                                              b will be replaced with "joint2"
        NOTE: The exception to this rule is the "decompose" kwarg, which changes the eval behavior.

        Args:
            expression (str): String of matrix multiplication expression.
            **kwargs:
                decompose (bool): If True, create a decomposeMatrix node for output to TRS channels.

        """
        # These have to be reset here because of the global var storing the class instance,
        # which I followed as the format from DGParser, presumably to save time from having to re-instance
        # the class and parser over and over.
        self.outputTarget = None
        self.matrixDict = {}
        self.parensDict = {}
        self.expression = expression
        self.mults = []

        decompose = kwargs.pop('decompose', False)
        self.kwargs = kwargs

        tokens = self.parser.parseString(expression, True)

        connect = True
        if all([t.endswith(f'.{self.getModifier}') for t in tokens]):
            connect = False

        result = self._evalTokens(tokens)

        if str(result) in self.matrixDict:
            result = self.matrixDict[result]

        if isinstance(result, OpenMaya.MMatrix):
            connect = False

        if not connect:
            if self.mults:
                cmds.delete(self.mults)

        if self.outputTarget:
            if connect:
                if decompose:
                    self.decomposeMatrix(self.outputTarget.split('.')[0], matrixPlug=result)
                else:
                    cmds.connectAttr(result, self.outputTarget, f=1)

            else:
                if '.' in self.outputTarget:
                    cmds.setAttr(self.outputTarget, *result, type='matrix')
                else:
                    cmds.xform(self.outputTarget.split('.')[0], ws=1, m=result)

        # Clean up orphaned mults
        for mult in self.mults[:-1]:
            if not cmds.listConnections(mult, s=0, d=1):
                cmds.delete(mult)

        return result

    def generateKey(self, length=8):
        """
        Generate a random alpha key for storing the various special items as we parse.

        Returns (str): random letter key

        """
        unique = ''.join([ascii_lowercase[random.randrange(0, 26)] for i in range(length)])
        return unique

    def storeOutput(self, tokens):
        """
        If the assignment operator "=" was used in the expression, store
        this plug for setting/connecting when finished.

        """
        if tokens:
            self.outputTarget, _ = self.getPlugAndModifiers(tokens.pop())

    def storeParens(self, tokens):
        unique = self.generateKey()
        self.parensDict[unique] = tokens[0]
        return unique

    def _storeMatrix(self, matrix):
        unique = self.generateKey()
        if not isinstance(matrix, OpenMaya.MMatrix):
            matrix = OpenMaya.MMatrix([float(i) for i in matrix])
        self.matrixDict[unique] = matrix
        return unique

    def storeMatrix(self, tokens):
        """
        Store a numerical matrix token list in class var,
        and replace it with a string that will be parsed properly.

        """
        if tokens:
            tokens[0] = self._storeMatrix(tokens)
            for i in range(1, len(tokens)):
                tokens.pop()

    def joinDots(self, tokens):
        """
        Replace any orphaned '.''s, which could result from evaluating
        expressions in parens first.

        This could maybe be taken care of in the parsing directly,
        but it was getting complicated, so path of least resistance.

        """
        tokenList = tokens.asList()
        while '.' in tokenList:
            index = tokenList.index('.')

            # Join the neighbors with the dot
            joined = ''.join(tokenList[index - 1:index + 2])
            tokenList.insert(index - 1, joined)

            # Pop the 3 items we replaced with the joined token
            tokenList.pop(index)
            tokenList.pop(index)
            tokenList.pop(index)

        return tokenList

    def _evalTokens(self, tokens):
        """
        Evaluate the token list of node.attr.modifiers

        Args:
            tokens (list): List of node.attr.modifiers*n strings

        Returns (str/OpenMaya.MMatrix): Resulting multMatrix node out plug, or resulting matrix.

        """
        mm = cmds.createNode('multMatrix', skipSelect=True)
        self.add_notes(mm)

        for i, nodeAttr in enumerate(tokens):
            multPlug = f'{mm}.matrixIn[{i}]'

            # Split the nodeAttr/kwarg from the modifiers
            nodePlug, modifiers = self.getPlugAndModifiers(nodeAttr)
            plugOrMatrix = nodePlug

            # Check if the string is referring to a stored parenthetical expression
            if plugOrMatrix in self.parensDict:
                plugOrMatrix = self._evalTokens(self.parensDict[plugOrMatrix])

            # Replace with stored matrix value
            if plugOrMatrix in self.matrixDict:
                plugOrMatrix = self.matrixDict[plugOrMatrix]

            # Check for the get modifier
            get = self.getModifier in modifiers

            # Apply modifiers
            if modifiers:
                for mod in modifiers:
                    if mod not in self.modifiers:
                        raise InvalidModifierException(f'Invalid modifier "{mod}"')

                    plugOrMatrix = self.modifiers[mod](plugOrMatrix, get=get)

                    if not get:
                        self.add_notes(plugOrMatrix.split('.')[0])

            # Set value, or connect plug to math node
            self.setOrConnect(plugOrMatrix, multPlug)

        # If all tokens have the '.G' suffix, we just care about the matrix value
        # and the mult can be deleted.  The string key is returned because this function
        # runs recursively if there are multiple nested parenthesis.
        if all([t.endswith(f'.{self.getModifier}') for t in tokens]):
            result = cmds.getAttr(f'{mm}.matrixSum')
            k = self._storeMatrix(result)
            cmds.delete(mm)
            return k

        # If we only had one token, no mult is taking place, so we just return the plug.
        elif len(tokens) == 1:
            self.disconnectPlug(plugOrMatrix, f'{mm}.matrixIn[0]')
            cmds.delete(mm)
            return plugOrMatrix

        self.mults.append(mm)
        return f'{mm}.matrixSum'

    def nodeAttrSplit(self, inputStr):
        tok = inputStr.split('.')
        return tok[0], tok[1]

    def inputToMatrix(self, input_):
        if isinstance(input_, OpenMaya.MMatrix):
            return input_
        elif isinstance(input_, str):
            return OpenMaya.MMatrix(cmds.getAttr(input_))

    def getPlugAndModifiers(self, inputStr):
        toks = inputStr.split('.')

        # replace with a kwarg assigned value
        if toks[0] in self.kwargs:
            replacedTok = self.kwargs[toks[0]]
            # The only instance where a kwarg is a list/tuple of these types should be a matrix.
            if isinstance(replacedTok, (OpenMaya.MMatrix, list, tuple)):
                replacedTok = self._storeMatrix(replacedTok)
            toks = replacedTok.split('.') + toks[1:]

        toksCopy = list(toks)
        i = 0
        # This takes the token string, and continually strips the last dot separated piece from it,
        # checking if the first chunk is one of:
        # 1. a maya node plug
        # 2. the key for a stored numerical matrix
        # 3. the key for a stored parens expression
        #
        # As soon as a condition is satisfied, returns the '.' joined string, and all tokens
        # that did not pass the condition check as the modifiers.
        while toks and not cmds.objExists('.'.join(toks)) \
                and ''.join(toks) not in self.matrixDict \
                and ''.join(toks) not in self.parensDict:
            toks = toks[:-1]
            i += 1
        return '.'.join(toks), toksCopy[len(toksCopy) - i:]

    # Modifiers
    # ---------------------------------------------------------------------------------
    def inverse(self, input_, get=False):
        if get:
            return self.inputToMatrix(input_).inverse()
        else:
            inv = cmds.createNode('inverseMatrix')
            cmds.connectAttr(input_, f'{inv}.inputMatrix')
            return f'{inv}.outputMatrix'

    def transpose(self, input_, get=False):
        if get:
            return self.inputToMatrix(input_).transpose()
        else:
            inv = cmds.createNode('transposeMatrix')
            cmds.connectAttr(input_, f'{inv}.inputMatrix')
            return f'{inv}.outputMatrix'

    def asMatrix(self, input_, get=False):
        if get:
            node, attr = self.nodeAttrSplit(input_)
            if attr == 'rotate' or attr == 'r':
                mat = rotate_to_matrix(cmds.getAttr(input_)[0], cmds.getAttr(f'{node}.ro'))
            elif attr == 'jointOrient' or attr == 'jo':
                mat = rotate_to_matrix(cmds.getAttr(input_)[0], 0)
            elif attr == 'translate' or attr == 't':
                mat = translate_to_matrix(cmds.getAttr(input_)[0])
            return mat
        else:
            comp = cmds.createNode('composeMatrix')
            if input_.endswith('rotate') or input_.endswith('jointOrient'):
                cmds.connectAttr(input_, f'{comp}.inputRotate')
            if input_.endswith('translate'):
                cmds.connectAttr(input_, f'{comp}.inputTranslate')
            return f'{comp}.outputMatrix'

    def pickT(self, input_, get=False):
        if get:
            mat = self.inputToMatrix(input_)
            mat = translate_to_matrix([mat[12], mat[13], mat[14]])
            return mat
        else:
            pick = cmds.createNode('pickMatrix', skipSelect=True)
            cmds.setAttr(f'{pick}.useRotate', 0)
            cmds.setAttr(f'{pick}.useScale', 0)
            cmds.setAttr(f'{pick}.useShear', 0)
            cmds.connectAttr(input_, f'{pick}.inputMatrix')
            return f'{pick}.outputMatrix'

    def pickR(self, input_, get=False):
        if get:
            mat = self.inputToMatrix(input_)
            mat.setElement(3, 0, 0)
            mat.setElement(3, 1, 0)
            mat.setElement(3, 2, 0)
            return mat
        else:
            pick = cmds.createNode('pickMatrix', skipSelect=True)
            cmds.setAttr(f'{pick}.useTranslate', 0)
            cmds.setAttr(f'{pick}.useScale', 0)
            cmds.setAttr(f'{pick}.useShear', 0)
            cmds.connectAttr(input_, f'{pick}.inputMatrix')
            return f'{pick}.outputMatrix'

    def get(self, input_, get=False):
        return self.inputToMatrix(input_)

    def decomposeMatrix(self, transform, matrixPlug=None, t=True, r=True, s=True):

        dcm = cmds.createNode('decomposeMatrix')
        if matrixPlug:
            cmds.connectAttr(matrixPlug, f'{dcm}.inputMatrix')

        if t:
            cmds.connectAttr(f'{dcm}.outputTranslate',
                         f'{transform}.translate', f=1)
        if r:
            cmds.connectAttr(f'{dcm}.outputRotate',
                         f'{transform}.rotate', f=1)
        if s:
            cmds.connectAttr(f'{dcm}.outputScale',
                         f'{transform}.scale', f=1)

        return f'{dcm}.inputMatrix'

    def setOrConnect(self, source, destination):
        if isinstance(source, OpenMaya.MMatrix):
            cmds.setAttr(destination, *source, type='matrix')
        else:
            cmds.connectAttr(source, destination)

    def disconnectPlug(self, source, destination):
        if not isinstance(source, OpenMaya.MMatrix):
            cmds.connectAttr(source, destination)

    def add_notes(self, node):
        node = node.split(".")[0]
        attrs = cmds.listAttr(node, ud=True) or []
        if "notes" not in attrs:
            cmds.addAttr(node, ln="notes", dt="string")
        keys = sorted(self.kwargs.keys())
        exp_kwargs_str = "\n  ".join([f"{x}: {self.kwargs[x]}" for x in keys])
        notes = f"Node generated by mpe\n\nExpression:\n  {self.expression}\n\nkwargs:\n  {exp_kwargs_str}"
        cmds.setAttr("{}.notes".format(node), notes, type="string")


def rotate_to_matrix(rot_vec, order):

    mat = OpenMaya.MEulerRotation(rot_vec, order=order).asMatrix()
    return mat


def translate_to_matrix(t_vector):
    mat = OpenMaya.MMatrix()
    mat.setElement(3, 0, t_vector[0])
    mat.setElement(3, 1, t_vector[1])
    mat.setElement(3, 2, t_vector[2])
    return mat