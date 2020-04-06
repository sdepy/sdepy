"""
================================
INFRASTRUCTURE FOR FUNCTIONS AND
CLASSES WITH MANAGED KEYWORDS
================================

*  ``kfunc`` decorator,
*  ``iskfunc`` test.
"""

import inspect
import warnings

from .integration import SDE
# (replace with SDE = type(None)
# to make this module stand alone)


########################################
#  Private functions for recurring tasks
########################################

def _wraps(wrapped):
    """
    Decorator to preserve some basic attributes
    when wrapping a function or class.
    """
    def decorator(wrapper):
        for attr in ('__module__', '__name__', '__qualname__', '__doc__'):
            setattr(wrapper, attr, getattr(wrapped, attr))
        return wrapper
    return decorator


##########################################
#  kfunc decorator: function wrapper with
#  managed keyword arguments
##########################################

def _kfunc_split_args(x, args):
    call_args = {k: z for k, z in args.items()
                 if k in x._kfunc_call_args}
    init_args = {k: z for k, z in args.items()
                 if k not in x._kfunc_call_args}
    return call_args, init_args


class _meta(type):
    """
    Metaclass that calls only __new__, not __init__,
    upon instantiation.
    """
    def __call__(cls, *var, **args):
        return cls.__new__(cls, *var, **args)


def _kfunc_decorate_class(f):
    """
    Decorator to wrap the given class as a kfunc.
    """

    # try to prevent unexpected collateral damage
    # with some preemptive checks
    # -------------------------------------------

    def checkattr(attr):
        return any(attr in vars(base) for base in f.__mro__[:-1])

    if not hasattr(f, '_is_kfunc'):
        # if f is not already a kfunc subclass, make sure
        # that __new__ has not been messed with,
        # that __init__ and __call__ have been provided,
        # and warn against params attribute overwriting
        if checkattr('__new__'):
            raise TypeError('class {} has a a customized __new__ method, '
                            'should not be wrapped as a kfunc'
                            .format(f))
        if not (checkattr('__init__') and checkattr('__call__')):
            raise TypeError('wrapping {} as a kfunc, no user defined '
                            '__init__ and/or __call__ methods were found'
                            .format(f))
        if hasattr(f, 'params'):
            warnings.warn('wrapping {} as a kfunc, the params attribute '
                          'will be overwritten'.format(f),
                          RuntimeWarning)

    # get init and call signatures and enforce discipline:
    # - init arguments always passed as keywords
    # - no overlapping between call and init keywords
    init_signature = [(k, z) for k, z in
                      inspect.signature(f.__init__).parameters.items()
                      if not z.kind == z.VAR_KEYWORD][1:]  # discard self
    call_signature = [(k, z) for k, z in
                      inspect.signature(f.__call__).parameters.items()
                      if not z.kind == z.VAR_KEYWORD][1:]  # discard self

    if any(z.kind != z.KEYWORD_ONLY for k, z in init_signature):
        raise TypeError(
            'cannot wrap {} as a kfunc, all of its parameters '
            '(initialization arguments) should be keyword-only'
            .format(f))

    init_signature = dict([(k, z.default) for k, z in init_signature])
    call_signature = dict(call_signature)
    if not set(call_signature).isdisjoint(init_signature):
        raise TypeError(
            'cannot wrap {} as a kfunc, none of its parameters '
            '(initialization arguments) should be named as any '
            'of its variables (calling arguments)'
            .format(f))

    # create and return a wrapper class as a subclass of f
    # ----------------------------------------------------
    @_wraps(f)
    class kfunc_class_wrapper(f, metaclass=_meta):
        """
        Decorator to add kfunc functionality to a class.
        A wrapping subclass of f is returned.
        """

        _kfunc_init_args = init_signature
        _kfunc_call_args = call_signature

        def __new__(cls, *call_vars, **args):

            # avoid unsafe subclassing of kfunc classes
            if hasattr(cls, '_is_kfunc') and '_is_kfunc' not in vars(cls):
                warnings.warn(
                    'to prevent unexpected init and call behavior, '
                    'a subclass of a kfunc class should be decorated '
                    'with kfunc, but {} was not'.format(cls),
                    RuntimeWarning)
                self = f.__new__(cls)
                self.__init__(*call_vars, **args)
                return self

            # separate function arguments from instantiation parameters
            call_args, init_args = _kfunc_split_args(cls, args)

            # create a new class instance
            self = object.__new__(cls)
            self._kfunc_params = init_args
            self._kfunc_parent = None
            self.__init__(**init_args)

            # either return the instance, or call it and
            # return the result
            if call_vars or call_args:
                return super(cls, self).__call__(*call_vars, **call_args)
            else:
                return self
        __new__.__wrapped__ = f.__init__

        @_wraps(f.__call__)
        def __call__(self, *call_vars, **args):

            # separate function arguments from instantiation parameters
            call_args, init_args = _kfunc_split_args(self, args)

            if not init_args:
                # if no parameters, make a plain call to self
                return super().__call__(*call_vars, **call_args)
            else:
                # merge stored parameters with current ones
                # (current parameters override stored ones)
                new_init_args = {**self._kfunc_params, **init_args}

                # instantiate a derived object with merged parameters
                cls = type(self)
                new = object.__new__(cls)
                new.__init__(**new_init_args)
                new._kfunc_params = new_init_args
                new._kfunc_parent = self

                # handle evaluation/instantiation
                # with merged parameters
                if call_vars or call_args:
                    return super(cls, new).__call__(*call_vars, **call_args)
                else:
                    return new
        __call__.__wrapped__ = f.__call__

        if issubclass(f, SDE):
            # For integrators, the kfunc.params property is specialized
            # to include all parameters used by the SDE
            # (access is read-only)
            @property
            def params(self):
                return {**self._kfunc_init_args,
                        **self._kfunc_params,
                        **self.args}
        else:
            @property
            def params(self):
                return {**self._kfunc_init_args,
                        **self._kfunc_params}

        # mark the class as wrapped as a kfunc
        _is_kfunc = True

    return kfunc_class_wrapper


def _kfunc_decorate_function(nvar):
    """
    Decorator to wrap the given function as a kfunc.
    """

    def decorator(f):
        if isinstance(f, type):
            raise SyntaxError('improper use of kfunc decorator - see '
                              'kfunc docstring')

        f_signature = [(k, z) for k, z in
                       inspect.signature(f).parameters.items()
                       if not z.kind == z.VAR_KEYWORD]
        if not 0 < nvar <= len(f_signature):
            # avoid unexpected behaviour with nvar <= 0
            raise ValueError('expecting 0 < nvar <= {}, not {}'
                             .format(len(f_signature), nvar))
        if any(z.kind != z.KEYWORD_ONLY for k, z in f_signature[nvar:]):
            raise TypeError(
                'error wrapping {} as a kfunc - expecting nvar={} '
                'initial keyword or non-keyword arguments, '
                'as kfunc variables, followed by keyword arguments only, '
                'as kfunc parameters'
                .format(f, nvar))

        f_init_args = dict([(k, z.default) for k, z in f_signature[nvar:]])
        f_call_args = set(dict(f_signature[:nvar]))

        @_kfunc_decorate_class
        @_wraps(f)
        class kfunc_function_wrapper:
            def __init__(self, **args):
                self._kfunc_params = args

            def __call__(self, *var, **args):
                return f(*var, **args, **self._kfunc_params)
            __call__.__wrapped__ = f

        kfunc_function_wrapper._kfunc_init_args = f_init_args
        kfunc_function_wrapper._kfunc_call_args = f_call_args
        kfunc_function_wrapper.__wrapped__ = f

        return kfunc_function_wrapper

    return decorator


def kfunc(f=None, *, nvar=None):
    """
    Decorator to wrap classes or functions as objects
    with managed keyword arguments.

    This decorator, intended as an aid to interactive and notebook sessions,
    wraps a callable, class or function, as a "kfunc" object
    that handles separately its parameters (keyword-only),
    whose values are stored in the object, and its variables
    (positional or keyword), always provided upon evaluation.

    Syntax::

        @kfunc
        class my_class:
            def __init___(self, **kwparams):
                ...
            def __call__(self, *var, **kwvar):
                ...

        @kfunc(nvar=k)
        def my_function(*var, **kwargs):
            ...

    After decoration, ``my_class`` is a kfunc with ``kwparams`` as
    parameters, and with ``var`` and ``kwvar`` as variables, and
    ``my_function`` is a kfunc with the first ``k`` of ``var, kwargs``
    as variables, and the remaining ``kwargs`` as parameters.
    For usage, see examples below.

    Attributes
    ----------
    params : dictionary
       Parameter values stored in the instance (read-only). For
       wrapped ``SDE`` subclasses, also includes default values
       of all SDE-specific parameters, as stored in the ``args`` attribute.

    Examples
    --------
    Wrap ``wiener_source`` into a kfunc, named ``dw``:

        >>> import numpy
        >>> from sdepy import wiener_source, kfunc
        >>> dw = kfunc(wiener_source)

    Instantiate ``dw`` and evaluate it (this is business as usual):

        >>> my_instance = dw(paths=100, dtype=numpy.float32)
        >>> x = my_instance(t=0, dt=1)
        >>> x.shape, x.dtype
        ((100,), dtype('float32'))

    Inspect kfunc parameters stored in ``my_instance``:

        >>> my_instance.params  # doctest: +SKIP
        {'paths': 100, 'vshape': (), 'dtype': <class 'numpy.float32'>, \
        'corr': None, 'rho': None}

    Evaluate ``my_instance`` changing some parameters (call
    the instance with one or more):

        >>> x = my_instance(t=0, dt=1, paths=999)
        >>> x.shape, x.dtype
        ((999,), dtype('float32'))

    Parameters stored in ``my_instance`` are not affected:

        >>> my_instance.paths == my_instance.params['paths'] == 100
        True

    Create a new instance, changing some parameters and keeping those
    already set (call the instance without passing any
    variables):

        >>> new_instance = my_instance(vshape=2, rho=0.5)
        >>> new_instance.params  # doctest: +SKIP
        {'paths': 100, 'vshape': 2, 'dtype': <class 'numpy.float32'>, \
        'corr': None, 'rho': 0.5}

    Instantiate and evaluate at once (pass one or more variables
    to the class constructor):

        >>> x = dw(0, 1, paths=100, dtype=numpy.float32)
        >>> x.shape, x.dtype
        ((100,), dtype('float32'))

    As long as variables are passed by name, order doesn't
    matter (omitted variables take default values, if any):

        >>> x = dw(paths=100, dtype=numpy.float32, dt=1, t=0)
        >>> x.shape, x.dtype
        ((100,), dtype('float32'))
    """
    # either f is a class and nvar is None,
    # or f is None and kfunc has been called with nvar
    # specified
    case1 = inspect.isclass(f) and nvar is None
    case2 = f is None and nvar is not None
    if not (case1 or case2):
        raise SyntaxError('improper use of kfunc decorator - see '
                          'kfunc docstring')
    if nvar is None:
        return _kfunc_decorate_class(f)
    else:
        return _kfunc_decorate_function(nvar)


def iskfunc(cls_or_object):
    """
    Tests if the given class or instance has been
    wrapped as a kfunc.
    """
    return hasattr(cls_or_object, '_is_kfunc')
