{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {%- if objname in ['PhysicsState', 'PhysicsTendency'] %}
      {%- set custom_methods = ['zeros', 'ones', 'copy', 'isnan', 'any_true'] %}
      {%- set filtered_methods = [] %}
      {%- for item in methods %}
         {%- if item in custom_methods %}
            {%- set _ = filtered_methods.append(item) %}
         {%- endif %}
      {%- endfor %}
      
      {% if filtered_methods %}
      .. rubric:: {{ _('Methods') }}

      .. autosummary::
      {% for item in filtered_methods %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
      {% endif %}
    {%- elif objname == 'Physics' -%}
      {%- set filtered_methods = [] -%}
      {%- for item in methods -%}
         {%- if item != '__init__' -%}
            {%- set _ = filtered_methods.append(item) -%}
         {%- endif -%}
      {%- endfor -%}
      
      {%- if filtered_methods %}
      .. rubric:: {{ _('Methods') }}

      .. autosummary::
      {% for item in filtered_methods %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
      {%- endif %}
   {%- else %}
      .. automethod:: __init__

      {% if methods %}
      .. rubric:: {{ _('Methods') }}

      .. autosummary::
      {% for item in methods %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
      {% endif %}
   {%- endif %}
   {% endblock %}

   {% block attributes %}
   {% endblock %}