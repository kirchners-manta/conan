{{ name | escape | underline}}


.. automodule:: {{ fullname }}

    {% block attributes %}
        {% if attributes %}
    .. rubric:: Module Attributes

    .. autosummary::
        :toctree:
        :template: attribute-template.rst
                {% for item in attributes %}
                    ~{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block functions %}
        {% if functions %}
    .. rubric:: Functions

    .. autosummary::
        :toctree:
        :template: function-template.rst
                {% for item in functions %}
                    ~{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block classes %}
        {% if classes|length == 1 %}
            {% for item in classes %}
    .. rubric:: {{ item }}

    .. _{{ item }}:
    .. autoclass:: {{ item }}
        :members:
        :private-members:
        :show-inheritance:

                    {% block methods %}

                        {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
                            {% for item in methods %}
                               ~{{ name }}.{{ item }}
                            {%- endfor %}

                        {% endif %}
                    {% endblock %}

            {%- endfor %}
        {% elif classes %}
    .. rubric:: Classes
    .. autosummary::
        :toctree:
        :template: class-template.rst
                {% for item in classes %}
                    ~{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}


    {% block exceptions %}
        {% if exceptions %}
    .. rubric:: {{ _('Exceptions') }}

    .. autosummary::
        :toctree:
                {% for item in exceptions %}
                    ~{{ item }}
                {%- endfor %}
                {% endif %}
    {% endblock %}

    {% block modules %}
        {% if  modules %}
    .. rubric:: Modules

    .. autosummary::
        :toctree:
        :template: module-template.rst
        :recursive:
                {% for item in modules %}
                ~{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}
