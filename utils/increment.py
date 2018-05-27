def preDecrement(name, local={}):
	"""
	Equivalent to --name
	:param name: name of the var
	:param local: namespace
	:return: Equivalent to --name
	"""
	if name in local:
		local[name] -= 1
		return local[name]
	globals()[name] -= 1
	return globals()[name]


def postDecrement(name, local={}):
	"""
	Equivalent to name--
	:param name: name of the var
	:param local: namespace
	:return: Equivalent to var--
	"""
	if name in local:
		local[name] -= 1
		return local[name] + 1
	globals()[name] -= 1
	return globals()[name] + 1


def preIncrement(name, local={}):
	"""
	Equivalent to ++name
	:param name: name of var
	:param local: namespace
	:return: Equivalent to ++var
	"""
	
	if name in local:
		local[name] += 1
		return local[name]
	globals()[name] += 1
	return globals()[name]


def postIncrement(name, local={}):
	"""
	Equivalent to name++
	:param name: name of var
	:param local: namespace
	:return: Equivalent to var++
	"""
	if name in local:
		local[name] += 1
		return local[name] - 1
	globals()[name] += 1
	return globals()[name] - 1
