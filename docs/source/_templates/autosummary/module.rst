{#
   Recursive module template, forked from Sphinx's default for one reason:
   to keep the test suite out of the public API tree.

   ``api.rst`` walks packages such as ``jcm.dycore`` with ``:recursive:``,
   and the default template forwards *every* submodule it finds. Because
   jcm co-locates its tests with their modules (``*_test.py`` beside the
   module, plus per-package ``conftest.py``), that walk imports the whole
   test suite at documentation-build time — which is both wrong (tests are
   not public API, and their stub pages were being published) and fragile:
   the pyses backend's test modules require the optional ``pyses`` extra,
   so a clean docs environment failed to import them and the strict build
   (``-W``) turned six such failures into errors (#829).

   Filtering here rather than mocking ``pyses`` keeps the build honest: the
   modules jcm actually documents are imported for real, so a genuinely
   broken import is still an error.
-#}
{%- set public_modules = [] -%}
{%- for item in modules -%}
   {%- set leaf = item.split('.')[-1] -%}
   {%- if not leaf.endswith('_test') and leaf != 'conftest' -%}
      {%- set _ = public_modules.append(item) -%}
   {%- endif -%}
{%- endfor -%}
{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if public_modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in public_modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
