import React, { useRef, useState } from "react";
import CanvasDraw from "react-canvas-draw";
import ButtonGroup from '../components/elements/ButtonGroup';
import Button from '../components/elements/Button';
import Input from '../components/elements/Input';
import '../assets/scss/canvas.css'

export const Demo = () => {

  const canvas = useRef()
  const [image, setImage] = useState()

  const reset = () => {
    canvas.current.clear()
  }

  const undo = () => {
    canvas.current.undo()
  }

  const uploadToAPI = (e) => {
    console.log(e.target.files)
    alert('this feature is not fully complete!')
  }

  return (
    <>
      <h1 className="mb-8 reveal-from-bottom upper-center" data-reveal-delay="200">
        Try It Yourself!
      </h1>
      <p className="mt-0 mb-16 reveal-from-bottom upper-center" data-reveal-delay="200"> 
        Write your equation, take a screenshot, and upload! Our API will do the rest. Write in consistent sizing for better results.
      </p>
      <CanvasDraw className='center' 
                  brushRadius={2} 
                  brushColor="#000000" 
                  canvasWidth={540} 
                  canvasHeight={360}
                  gridColor="rgba(150,150,150,0)"
                  hideInterface={true}
                  lazyRadius={0}
                  ref={canvas}
                  />
        <div className="lower-center" data-reveal-delay="600">
          <ButtonGroup>
            <Button tag="a" color="dark" wideMobile onClick={reset}>
              Reset Canvas
            </Button>
            <Button tag="a" color="dark" wideMobile onClick={undo}>
              Undo Stroke
            </Button>
            <input tag="a" type="file" name="file" onClick={uploadToAPI}/>
          </ButtonGroup>
        </div>
    </>
  );
}

