var value = '';
var keywords = [];
var emphasized_keywords = [];
var query_val, num_products;

$('input#query').on('keyup', function() {
  value = $(this).val();
});

$('.emphasize').click(function(){
  emphasized_keywords = [];
  keywordsDiv = $(this).next();
  keywordsDiv.empty();
  keywordsDiv.css("display", "block");
  keywords = value.split(/[ ,]+/);
  for (i = 0; i < keywords.length; i++) {
    $( "<p class = 'keyword'>"+keywords[i]+"</p>" ).appendTo(keywordsDiv);
  };
});

$(document).on('click', '.keyword', function(){
  $this = $(this)
  thisText = $this.text();
  $emphasize = $('#emphasize')
  if ($this.hasClass('active')){
    $this.css({'color':'#ccc'});
    $this.removeClass('active');
    var index = emphasized_keywords.indexOf(thisText);
    if (index >= 0) emphasized_keywords.splice(index,1);
  }else{
    $this.css({'color':'#000'});
    $this.addClass('active');
    emphasized_keywords.push(thisText);
  };
  $emphasize.val(emphasized_keywords);
});

function searchFormSubmit(index = 0){
  console.log(index);

  var form = $('#search_form');
  var query = $("#query");
  var emphasized_keywords = $("#emphasize");
  var first = $('#first');
  var container = $('#container');
  container.css({'display':'none'});

  console.log(emphasized_keywords.val());
  if(emphasized_keywords.val() == '') $('.keywords').css('display', 'none');

  $( "<p id = 'loadingMsg'>LOADING THE PRODUCT AND ITS REVIEWS</p>").
  appendTo($('#search_form')).
  css({'display':'block', 'text-align':'left'});

  setInterval(function () {
      $('#loadingMsg').text($('#loadingMsg').text() + ".");
  },1000);

    $.ajax({
        type: 'POST',
        url: '/predict',
        data: JSON.stringify({"query" : query.val(), "emphasized_keywords": emphasized_keywords.val(), "index": index, "first": first.val()}),
        contentType: "application/json; charset=utf-8",
        dataType: 'html',
        cache: false,
        success: function(data){
          $('#loadingMsg').remove();
          var results = $(data).filter('#results').text();
          container.empty();
          container.css({'display':'block'});
          container.append(results);
          $('.row_heading').remove(); $('.blank').remove(); $('.col4').css('display', 'none');
          query.attr('placeholder', query_val);
          query.val('');
          if(parseInt(first.val())){
            num_products = parseInt($('#num_products').text());
            emphasized_keywords.val('');
            $('.keywords').css('display', 'none');
            first.val('0');
          }
          emphasized_keywords.val('');
        },error: function(xhr, ajaxOptions, thrownError){
            return;
        }
    });
}


$("#search_form").submit(function(e) {
    e.preventDefault();
    query_val = $('#query').val();
    $('#first').val('1');
    searchFormSubmit(0);
});

$(document).on('click', 'td.col0', function(){
  $this = $(this);
  originalText = $this.siblings(':last').text();
  thisText = $this.text();
  if((originalText != 'nan') & (thisText != 'nan')){
    $this.text(originalText);
    $this.siblings(":last").text(thisText);
  }
});

$(document).on('mouseenter', 'td.col0', function(){
    $this = $(this);
    originalText = $this.siblings(':last').text();
    thisText = $this.text();
    if((originalText != 'nan') & (thisText != 'nan')) $this.css({'cursor':'pointer','text-decoration':'underline'});
});

$(document).on('mouseleave', 'td.col0', function(){
    $(this).css({'text-decoration':'none'});
});

$(document).on('click', '#before', function(e){
    e.preventDefault();
    $('#first').val('0');
    var index_val = parseInt($('#index').val());
    if(index_val > 0){
        index_val = index_val - 1
        $('#index').val(index_val);
        searchFormSubmit(index_val);
    }else{
        alert('This is the first product we suggest.')
    }
});

$(document).on('click', '#after', function(e){
    e.preventDefault();
    $('#first').val('0');
    var index_val = parseInt($('#index').val());
    if(index_val < num_products - 1){
        index_val = index_val + 1
        $('#index').val(index_val);
        searchFormSubmit(index_val);
    }else{
        alert('Sorry, this is the last product we suggest.')
    }
});