$(document).ready(function() {

    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie != '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) == (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
            return cookieValue;
    }
    var csrftoken = getCookie('csrftoken'); 

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
        }
        $.ajaxSetup({
            beforeSend: function(xhr, settings) {
                if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                    xhr.setRequestHeader("X-CSRFToken", csrftoken);
                }
            }
        });
    
    // Select all li items whose class begins with "diff". These are the problems.
    // Pass them to autorender
    $('[class^="diff"]').each( function () {
        renderMathInElement( $(this)[0] );
    });

    // adds a datepicker jquery ui element to dates
    //$('#id_expires').datepicker()
    //$('#id_live').datepicker()
    jQuery('#id_expires').datetimepicker(
            { format:'Y-m-d H:i',
            });
    jQuery('#id_live').datetimepicker(
            { format:'Y-m-d H:i',
            });
    

    // On Problem set page, deal with ability to print questions and solutions.
    // First, nice JS for "all" checkboxes. Unnecessary but nice.
    $("[id$='-all']").click( function() {
        qs=$(this).attr('id').split('')[0]
        $("[name^='"+qs+"-']").prop("checked", $(this).is(":checked"));
    });

    // Cannot include solution without text
    $("[name^='s-']").click( function() {
        var thisNum = $(this).attr('name').split('-')[1];
        $("[name='q-"+thisNum+"']").prop("checked",true);
    });

    $('span.choice-remove').click( function() {
        data_id = $(this).attr('data-id');
        $('[data-id='+data_id+']').remove();
        $('form').submit();
    });

    if ($("#id_q_type option:selected").text() != "Multiple Choice") {
        $("#id_mc_choices").prop("disabled", true);
    }

    $("#id_q_type").change( function() {
        if ($(this).find("option:selected").text() == 'Multiple Choice') {
            $("#id_mc_choices").prop('disabled', false);
        } else {
            $("#id_mc_choices").prop('disabled', true);
        }
    });

});
