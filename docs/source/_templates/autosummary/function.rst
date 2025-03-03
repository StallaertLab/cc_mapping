.. py:function:: {{ fullname.split('.')[-2] + '.' + fullname.split('.')[-1] }}
   :noindex:

   {% if objname == fullname %}
   .. automodule:: {{ module }}
      :members: {{ objname.split('.')[-1] }}
   {% else %}
   .. autoclass:: {{ module }}.{{ objname.split('.')[-1] }}
      :members:
   {% endif %}
