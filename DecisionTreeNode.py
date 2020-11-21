from anytree import NodeMixin, RenderTree


class DecisionTreeNode(NodeMixin):

    def __init__(self, data, criterion=None, criterion_value=None, decision_rule=None, parent=None, children=None):
        """A Node class for a DecisionTree using the anytree library.

        Args:
            data (pandas.DataFrame): A DataFrame containing the (train)-data to fit the DecisionTreeNode. The DataFrame must include the class attribute.
            criterion (string): Represents a measure for the impurity of the data. Supported are information gain, gini index and misclassification error.
            criterion_value (string): Represents the impurity of the given node measured by criterion.
            decision_rule ((string, string)): decision rule expression for the given attribute. first value of tuple is attribute second is decision rule expression (e.g. '==', '<' and '>=')
            parent (NodeMixin, optional): The parent node. Can be any arbitrary instance of NodeMixin. If None, node is root node
            children ([NodeMixin], optional): The chidlren nodes. Can be any arbitrary iterable of NodeMixin. if none, node is leaf
        """
        super(DecisionTreeNode, self).__init__()
        self.data = data
        self.criterion = criterion
        self.decision_rule = decision_rule
        self.parent = parent
        if children:
            self.children = children


tree = DecisionTreeNode('hello')
left = DecisionTreeNode('left', parent=tree)
right = DecisionTreeNode('right', parent=tree)

for pre, _, node in RenderTree(tree):
    treestr = u"%s%s" % (pre, node.data)
    print(treestr.ljust(8), node.data)
