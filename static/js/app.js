function classify() {
	var t0 = performance.now();

  var data = {"text": document.getElementById("reviewText").value}
  console.log("Now i will call python file with input" + data);	

  $.post("/classify", data, function(data, status){
	  console.log("This code will be executed");
    var img = document.getElementById('sentimentImg');
    if (data.probability == 1) {
      img.setAttribute('src', 'static/img/happy.svg');
    } else {
      img.setAttribute('src', 'static/img/sad.svg');
    }
    document.getElementById('probText').textContent = '(' + data.sentiment + ' with probability: ' + data.probability + '%)'
  }, "json");
  var t1 = performance.now();
  alert("Time taken for classification took " + (t1 - t0) + " milliseconds.")
}


// Prevent default submit behaviour
$("#reviews_form").submit(function(e) {
    e.preventDefault();
});
