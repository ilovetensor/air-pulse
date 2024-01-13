html_code = """
<div style="background: transparent;">
  <h1>Breathe Smarter:<br> Know About Your
    <span class='location'></span>'s 
    <span class='time-period'></span> Air
  </h1>
</div>

<style>
  @import url('https://fonts.googleapis.com/css?family=Work+Sans:900');

  div {
    font-family:  system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    background: rgba(0,0,0,0);
  }
  * {
    background: rgba(0,0,0,0);
  }

  div {
    text-align: left;
    display: flex;
    flex-direction: column;
    max-width: 1000px;
    margin: 0;
   
    font-size: 20px;
    letter-spacing: 1px;
    color: #001d3d;
    animation-duration: 3s;
  }

  .location, .time-period {
    display: inline-block;
    text-transform: ;
    color: white;
    letter-spacing: 3px;
    width: auto;
    animation: slidedown 3s infinite;
    -webkit-animation: slidedown 3s infinite;
    -moz-animation: slidedown 3s infinite;
  }

  @keyframes slidedown {
    0% {
      opacity: 50;
    }

    5% {
      opacity: 0;
      transform: translateY(0.5em);
    }

    7% {
      transform: translateY(-1em);
    }

    10% {
      opacity: 0;
    }

    20% {
      transform: translateY(0);
      opacity: 100;
    }
  }
</style>

<script>
  let i = 0;
  let k = 1;

  const randomizeText = () => {
    const locationElement = document.querySelector('.location');
    const timePeriodElement = document.querySelector('.time-period');

    const compStylesLocation = window.getComputedStyle(locationElement);
    const compStylesTimePeriod = window.getComputedStyle(timePeriodElement);

    const animationLocation = compStylesLocation.getPropertyValue('animation');
    const animationTimePeriod = compStylesTimePeriod.getPropertyValue('animation');

    const animationTimeLocation = parseFloat(animationLocation.match(/\d*[.]?\d+/)) * 00;
    const animationTimeTimePeriod = parseFloat(animationTimePeriod.match(/\d*[.]?\d+/)) * 1000;

    const locations = ['City', 'State', 'Country'];
    const timePeriods = ['Past', 'Present', 'Future'];

    i = randomNum(i, locations.length);
    const newLocation = locations[i];

    k = randomNum(k, timePeriods.length);
    const newTimePeriod = timePeriods[k];

    setTimeout(() => {
      locationElement.textContent = newLocation;
    }, Math.max(animationTimeLocation, animationTimeTimePeriod)); // Change duration for location change

    setTimeout(() => {
      timePeriodElement.textContent = newTimePeriod;
    },Math.max(animationTimeLocation, animationTimeTimePeriod));  // Change duration for time period change
  }

  const randomNum = (num, max) => {
    let j = Math.floor(Math.random() * max);

    // ensure diff num every time
    if (num === j) {
      return randomNum(i, max);
    } else {
      return j;
    }
  }

  const getAnimationTime = () => {
    const locationElement = document.querySelector('.location');
    const timePeriodElement = document.querySelector('.time-period');

    const compStylesLocation = window.getComputedStyle(locationElement);
    const compStylesTimePeriod = window.getComputedStyle(timePeriodElement);

    let animationLocation = compStylesLocation.getPropertyValue('animation');
    let animationTimePeriod = compStylesTimePeriod.getPropertyValue('animation');

    // firefox support for non-shorthand property
    animationLocation != "" ? animationLocation : animationLocation = compStylesLocation.getPropertyValue('-moz-animation-duration');
    animationTimePeriod != "" ? animationTimePeriod : animationTimePeriod = compStylesTimePeriod.getPropertyValue('-moz-animation-duration');

    // webkit support for non-shorthand property
    animationLocation != "" ? animationLocation : animationLocation = compStylesLocation.getPropertyValue('-webkit-animation-duration');
    animationTimePeriod != "" ? animationTimePeriod : animationTimePeriod = compStylesTimePeriod.getPropertyValue('-webkit-animation-duration');

    const animationTimeLocation = parseFloat(animationLocation.match(/\d*[.]?\d+/)) * 1000;
    const animationTimeTimePeriod = parseFloat(animationTimePeriod.match(/\d*[.]?\d+/)) * 1000;

    return Math.max(animationTimeLocation, animationTimeTimePeriod);
  }

  randomizeText();
  setInterval(randomizeText, getAnimationTime());
</script>
"""