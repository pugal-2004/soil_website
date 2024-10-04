import React from 'react';
import { Link, useNavigate } from 'react-router-dom'; // Import useNavigate from react-router-dom
import './FarmBuddy.css';

const Farmbuddy = () => {
  const navigate = useNavigate(); // Get the navigate function

  const handleExploreClick = () => {
    navigate('/new-page'); // Navigate to the new page
  };

  return (
    <>
      <header className="header">
        <a href="#" className="logo">
          <i className="fas fa-seedling"></i> FarmForesight
        </a>
        <nav className="navbar">
          <a href="#home">
            <i className="fas fa-home"></i> HOME
          </a>
          <a href="#features">
            <i className="fas fa-file-alt"></i> FEATURES
          </a>
          <a href="#categories">
            <i className="fas fa-th-large"></i> HELP
          </a>
          <a href="#about">
            <i className="fas fa-envelope"></i> QUERY
          </a>
        </nav>
      </header>

      <section className="home" id="home">
        <div className="video-background">
          <video autoPlay muted loop>
            <source src="farming_video.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
        <div className="content bg-green-600 w-full p-96 items-center">
          <h3><span style={{ color: 'azure' }}>fast</span> and <span style={{ color: 'azure' }}>perfect</span> crop recommender for soil</h3>
          <p>Find perfect crops for your soil</p>
          <div id="notification-container">
            <button
              id="explore-button"
              style={{ padding: '10px', backgroundColor: 'green', color: 'aliceblue', borderRadius: '20px' }}
              onClick={handleExploreClick}
            >
              EXPLORE
            </button>
          </div>
        </div>
      </section>

      <section className="features" id="features">
        <h1 className="heading">our <span>features</span></h1>
        <div className="box-container">
          <div className="box">
            <img src="https://th.bing.com/th/id/OIP.sMQXhCmQtqgzCJwolkQO1QHaEL?w=276&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7" alt="" />
            <h3>find crops for your soil</h3>
            <p>Here, we help to find crops for your soil</p>
          </div>
          <div className="box">
            <img src="" alt="" />
            <h3>pesticide/fertilizer recommendation</h3>
            <p>We will suggest you pesticides and fertilizers for your crops</p>
          </div>
          <div className="box">
            <img src="" alt="" />
            <h3>get information on state basis</h3>
            <p>Get all information on a state basis for better results</p>
          </div>
        </div>
      </section>

      <section className="categories" id="categories">
        <h1 className="heading">how to <span>use</span></h1>
        <div className="box-container">
          <div className="box">
            <img src="https://thumbs.dreamstime.com/b/explore-word-written-sign-board-vector-illustration-259864720.jpg" alt="" />
            <h3>explore option</h3>
            <p>give your soil input for your crop recommendation</p>
          </div>
          <div className="box">
            <img src="https://miro.medium.com/v2/resize:fit:1200/0*M1p7RZcb8Tt6kULL.png" alt="" />
            <h3>database history</h3>
            <p>can see your all crop recommendations history by pressing three horizontal lines on right bottom</p>
          </div>
          <div className="box">
            <img src="https://th.bing.com/th/id/OIP.N74Nrt6mRCR_MBeBXHFAyQHaFI?rs=1&pid=ImgDetMain" alt="" />
            <h3>contact us</h3>
            <p>contact through mail or raise a query by contact us button on end of the website</p>
          </div>
        </div>
      </section>

      <section className="about" id="about">
        <h1 className="heading">Raise <span>Query</span></h1>
        <div className="swiper about-slider">
          <div className="swiper-wrapper">
            <div className="swiper-slide box">
              <img src="https://duhocvip.com/wp-content/uploads/2017/06/contact-us.jpg" alt="" />
              <p>Feel free to contact us any time and get a response within 2 working days via email!</p>
              <a href="mailto:yogaarun67@gmail.com">
                <button style={{ padding: '10px', borderRadius: '20px', backgroundColor: 'green', color: 'aliceblue' }}>Contact us</button>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Sticky Hamburger Icon */}
      <Link to="/new-page" className="hamburger-icon">
        <i className="fas fa-bars"></i>
      </Link>
    </>
  );
};

export default Farmbuddy;
